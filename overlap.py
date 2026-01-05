import cv2
import numpy as np
import os, re, math
from collections import defaultdict

class OverlapDetector:
    def __init__(self, resize=0, threshold=0.4):
        self.resize = resize
        self.threshold = threshold
    
    @staticmethod
    def load_images(images_paths):
        list_images = []
        valid_paths = []
        for image_path in images_paths:
            image = cv2.imread(image_path)
            if image is None:
                continue
            h, w = image.shape[:2]
            if h > w:
                image = image[:1280, :]
            else:
                image = image[:720, :]
            list_images.append(image)
            valid_paths.append(image_path)
        return list_images, valid_paths

    @staticmethod
    def find_features(image, mode="SIFT"):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if mode == "SIFT":
            features = cv2.SIFT_create()
        else:
            features = cv2.AKAZE_create()
        keypoints, descriptors = features.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) == 0:
            descriptors = np.empty((0, 128), dtype=np.float32)
            keypoints = []
        else:
            descriptors = descriptors.astype(np.float32, copy=False)
        return keypoints, descriptors

    @staticmethod
    def match_features(desc1, desc2, ratio=0.75):
        if desc1 is None or desc2 is None or desc1.size == 0 or desc2.size == 0:
            return []
        is_binary = desc1.dtype == np.uint8
        if is_binary:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        try:
            raw_matches = matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []
        good_matches = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)

        return good_matches

    @staticmethod
    def filter_by_angle_cosine(kp1, kp2, matches, max_angle_deg=45):
        if not matches:
            return []
        max_cos = math.cos(math.radians(max_angle_deg))  # cos(45) = 0.707
        filtered = []
        for m in matches:
            pt1 = np.array(kp1[m.queryIdx].pt)
            pt2 = np.array(kp2[m.trainIdx].pt)
            vec = pt2 - pt1
            norm = np.linalg.norm(vec)
            if norm < 1e-5:
                continue
            unit_vec = vec / norm
            cos_angle = unit_vec[0]
            if abs(cos_angle) >= max_cos:
                filtered.append(m)
        # print('DEBUG DEG: ', max_angle_deg)
        return filtered

    @staticmethod
    def apply_mask(img, mask):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.where(mask[..., None] == 255, img, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    
    @staticmethod
    def compute_overlap(src, dst, H, prefix="overlap"):
        if H is None or not isinstance(H, np.ndarray) or H.shape != (3, 3):
            return None

        H = H.astype(np.float32, copy=False)
        if not np.isfinite(H).all():
            return None

        det, H_inv64 = cv2.invert(H, flags=cv2.DECOMP_LU)
        if abs(float(det)) < 1e-12:
            return None
        H_inv = H_inv64.astype(np.float32, copy=False)
        if not np.isfinite(H_inv).all():
            return None

        h_src, w_src = src.shape[:2]
        h_dst, w_dst = dst.shape[:2]

        corners_src = np.float32([[0, 0], [0, h_src - 1], [w_src - 1, h_src - 1], [w_src - 1, 0]]).reshape(-1, 1, 2)
        dst_corners_from_src = cv2.perspectiveTransform(corners_src, H)

        warp_mask_dst = np.zeros((h_dst, w_dst), dtype=np.uint8)
        cv2.fillConvexPoly(warp_mask_dst, np.int32(dst_corners_from_src), 255)
        dst_mask = np.full((h_dst, w_dst), 255, dtype=np.uint8)

        intersection = cv2.bitwise_and(warp_mask_dst, dst_mask)
        union = cv2.bitwise_or(warp_mask_dst, dst_mask)
        union_sum = int(np.count_nonzero(union))
        overlap_ratio = float(np.count_nonzero(intersection) / union_sum) if union_sum > 0 else 0.0

        src_overlap_mask = cv2.warpPerspective(
            intersection, H_inv, (w_src, h_src),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        src_result = OverlapDetector.apply_mask(src, src_overlap_mask)
        dst_result = OverlapDetector.apply_mask(dst, intersection)

        poly_src, poly_dst = [], []
        for mask, res, poly_list in [
            (intersection, dst_result, poly_dst),
            (src_overlap_mask, src_result, poly_src)
        ]:
            if np.count_nonzero(mask) > 0:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
                    cv2.polylines(res, [approx], True, (0, 255, 0), 2)
                    poly_list.append(approx.reshape(-1, 2).tolist())

        if union_sum == 0 or overlap_ratio == 0:
            return None

        cv2.imwrite(f"tmp/{prefix}_src_overlap.jpg", src_result)
        cv2.imwrite(f"tmp/{prefix}_dst_overlap.jpg", dst_result)

        return {
            "overlap_ratio": overlap_ratio,
            "src_polys": poly_src,
            "dst_polys": poly_dst,
        }
    
    @staticmethod
    def draw_matches(src, dst, kp1, kp2, matches, H, output_path="tmp/matches.jpg"):
        if H is None or not kp1 or not kp2 or not matches:
            return
        h1, w1 = src.shape[:2]
        h2, w2 = dst.shape[:2]
        corners = np.float32([[0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0]]).reshape(-1,1,2)
        poly = cv2.perspectiveTransform(corners, H)
        intersection = cv2.bitwise_and(
            cv2.fillConvexPoly(np.zeros((h2, w2), dtype=np.uint8), np.int32(poly), 255),
            np.uint8(np.any(dst > 0, axis=-1) * 255)
        )
        src_mask = cv2.warpPerspective(intersection, np.linalg.inv(H), (w1, h1))

        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = OverlapDetector.apply_mask(src, src_mask)
        vis[:h2, w1:] = OverlapDetector.apply_mask(dst, intersection)

        for m in matches:
            pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
            pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([w1, 0]))
            if 0 <= pt1[0] < w1 and 0 <= pt1[1] < h1 and 0 <= (pt2[0]-w1) < w2 and 0 <= pt2[1] < h2:
                if src_mask[pt1[1], pt1[0]] == 255 and intersection[pt2[1], pt2[0] - w1] == 255:
                    cv2.circle(vis, pt1, 2, (0, 255, 0), -1)
                    cv2.circle(vis, pt2, 2, (0, 255, 0), -1)
                    cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
        cv2.imwrite(output_path, vis)

    def estimate_homography(self, src, dst, ransac_thresh=8.0, overlap_thresh=0):
        if src.shape[0] != dst.shape[0]:
            kp1, desc1 = self.find_features(src, mode="AKAZE")
            kp2, desc2 = self.find_features(dst, mode="AKAZE")
            matches = self.match_features(desc1, desc2, ratio=0.55)
            matches = self.filter_by_angle_cosine(kp1, kp2, matches, max_angle_deg=65) # 65
        elif src.shape[0] == dst.shape[0] and src.shape[0] < src.shape[1]:
            kp1, desc1 = self.find_features(src, mode="AKAZE")
            kp2, desc2 = self.find_features(dst, mode="AKAZE")
            matches = self.match_features(desc1, desc2, ratio=0.7)
            matches = self.filter_by_angle_cosine(kp1, kp2, matches, max_angle_deg=45) # 45
        else:
            kp1, desc1 = self.find_features(src, mode="AKAZE")
            kp2, desc2 = self.find_features(dst, mode="AKAZE")
            matches = self.match_features(desc1, desc2, ratio=0.7)
            matches = self.filter_by_angle_cosine(kp1, kp2, matches, max_angle_deg=40) # 50
        # print("DEBUG LEN MACHES: ", len(matches))
        if len(matches) < 12:
            return None, None, None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

        overlap_result = self.compute_overlap(src, dst, H)
        # self.draw_matches(src, dst, kp1, kp2, matches, H)
        if overlap_result is None or overlap_result["overlap_ratio"] < overlap_thresh:
            return None, None, None
        return H, mask, overlap_result
    
    def compare(self, images):
        overlap_pairs = []
        l = len(images)
        conf_mat = np.full((l, l), np.nan, dtype=np.float32)

        for i in range(l):
            for j in range(i + 1, l):
                H, mask, res = self.estimate_homography(images[i], images[j])  # i → j

                if res is None or "overlap_ratio" not in res:
                    continue

                r = round(res["overlap_ratio"], 2)
                src_polys = res.get("src_polys", [])
                dst_polys = res.get("dst_polys", [])

                conf_mat[i, j] = r

                overlap_pairs.append({
                    "src_index": i,
                    "dst_index": j,
                    "src_polys": src_polys,
                    "dst_polys": dst_polys,
                    "overlap_ratio": float(r),
                })

        return conf_mat, overlap_pairs

    def overlap_detector(self, images_path):
        imgs, valid_pths = self.load_images(images_path)
        conf_mat, overlap_pairs = self.compare(imgs)
        i, j = np.triu_indices_from(conf_mat, k=1)
        mask = conf_mat[i, j] > self.threshold
        overlap_indxes = sorted(set(i[mask]) | set(j[mask]))
        print(conf_mat)
        return overlap_pairs, valid_pths
    
    def polygon_overlap_ratio(self, poly, box):
        contour = np.array(poly, dtype=np.int32)
        box_pts = np.array([
            [box.x1, box.y1],
            [box.x2, box.y1],
            [box.x2, box.y2],
            [box.x1, box.y2]
        ], dtype=np.int32)

        xmin = min(contour[:, 0].min(), box_pts[:, 0].min())
        ymin = min(contour[:, 1].min(), box_pts[:, 1].min())
        xmax = max(contour[:, 0].max(), box_pts[:, 0].max())
        ymax = max(contour[:, 1].max(), box_pts[:, 1].max())

        w = xmax - xmin + 1
        h = ymax - ymin + 1

        mask_poly = np.zeros((h, w), dtype=np.uint8)
        mask_box = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_poly, [contour - [xmin, ymin]], 1)
        cv2.fillPoly(mask_box, [box_pts - [xmin, ymin]], 1)

        inter = np.logical_and(mask_poly, mask_box).sum()
        box_area = mask_box.sum()

        if box_area == 0:
            return 0.0
        return inter / box_area
    
    def is_valid_polygon(self, poly):
        if isinstance(poly, list) and len(poly) == 1 and isinstance(poly[0], list):
            poly = poly[0]
        return isinstance(poly, list) and len(poly) >= 4

    def filter_overlap_products(self, images_info):
        image_paths = [image_info["image_path"] for image_info in images_info if image_info["valid"]]
        overlap_pairs, idx_images_overlap = self.overlap_detector(image_paths)
        inside_overlap = False
        if overlap_pairs:
            overlap_ratio = max((item["overlap_ratio"] if item.get("overlap_ratio") is not None and not np.isnan(item["overlap_ratio"]) else 0.0) for item in overlap_pairs)
        else:
            overlap_ratio = 0.0
        # Gom bbox overlap theo ảnh 1
        overlaps_by_image = defaultdict(list)
        for item in (overlap_pairs or []):
            # print("DEBUG ITEM: ", item)
            dst_path = image_paths[item["dst_index"]]
            for poly in item.get("dst_polys", []):
                overlaps_by_image[dst_path].append(poly)

        overlap_area_infos = []
        for img_path, polys in overlaps_by_image.items():
            for poly in polys:
                if self.is_valid_polygon(poly) and overlap_ratio >= 0.02:
                    overlap_area_infos.append((img_path, poly))

        filtered_boxes = []
        for image_info in images_info:
            image_info["overlap_boxes"] = []
            img_path = image_info["image_path"]
            for box in image_info.get("boxes", []):
                if box.class_name == 'Overlap_areas':
                    continue
                inside_overlap = False
                for overlap_img_path, poly in overlap_area_infos:
                    if img_path != overlap_img_path:
                        continue
                    box_ratio = self.polygon_overlap_ratio(poly, box)
                    if box_ratio > 0.35:
                        inside_overlap = True
                        image_info["overlap_boxes"].append(box)
                        break
                if not inside_overlap:
                    filtered_boxes.append(box)

            # Lọc boxes floor_separation
            filtered_floor_separation = {}
            for floor_name, floor_boxes in image_info.get("floor_separation", {}).items():
                filtered_floor_boxes = []
                for box in floor_boxes:
                    inside_overlap = False
                    for overlap_img_path, poly in overlap_area_infos:
                        if img_path != overlap_img_path:
                            continue
                        box_fl_ratio = self.polygon_overlap_ratio(poly, box)
                        if box_fl_ratio > 0.35:
                            inside_overlap = True
                            break
                    if not inside_overlap:
                        filtered_floor_boxes.append(box)
                filtered_floor_separation[floor_name] = filtered_floor_boxes
            image_info["floor_separation"] = filtered_floor_separation

        for i, image_info in enumerate(images_info):
            images_info[i]["overlap_area"] = []
            for image_path, poly in overlap_area_infos:
                if image_info["image_path"] == image_path:
                    images_info[i]["overlap_area"].append(poly)
        # print("DEBUG 1: ", images_info)
        return filtered_boxes, images_info, {
            "inside_overlap": inside_overlap,
            "overlap_ratio":overlap_ratio,
            "idx_images_overlap": idx_images_overlap}

if __name__ == "__main__":
    detector = OverlapDetector()
    image_paths = [
        r"tmp\img1.jpeg",
        r"tmp\img2.jpeg"
    ]
    overlap_pairs, pths = detector.overlap_detector(image_paths)
    print(overlap_pairs)