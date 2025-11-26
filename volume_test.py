# -*- coding: utf-8 -*-
r"""
.\.venv\Scripts\Activate.ps1

(접시 지름 스케일 + Perspective 보정 + 평면 피팅)을 적용하여
Depth Pro 깊이맵 + YOLOv8-seg 마스크로 음식의 부피(ml)와 질량(g)을 계산하는 스크립트.

실행 예시:
  데모 (합성 데이터):
    python volume_test.py

  YOLO + 실제 데이터:
    python volume_test.py --mode yolo --image ./scene.jpg --depth ./depth.npy --plate-cm 26 --yolo-weights yolov8x-seg.pt
"""

import argparse
import sys
import numpy as np
import cv2

# YOLO 사용 여부를 동적으로 결정
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    _YOLO_AVAILABLE = False


# ---------------------------------------------------------
# [유틸] 접시 지름(픽셀) 추정
# ---------------------------------------------------------
def estimate_plate_diameter_px(plate_mask: np.ndarray) -> int:
    H, W = plate_mask.shape
    max_width = 0
    # 간단히 행 단위로 스캔하여 최대 폭을 찾음
    # (실제로는 cv2.minAreaRect가 더 강건하지만, 원본 로직 유지)
    for y in range(H):
        xs = np.where(plate_mask[y])[0]
        if xs.size:
            width = xs.max() - xs.min() + 1
            if width > max_width:
                max_width = width
    
    # 만약 위의 방식으로 못 찾았다면(mask가 끊겨있거나 등), 윤곽선 기반으로 재시도
    if max_width == 0:
        contours, _ = cv2.findContours(plate_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest)
            max_width = int(max(rect[1])) # width, height 중 큰 값
            
    return int(max_width)


# ---------------------------------------------------------
# [유틸] 합성 데이터 생성 (Depth Pro 형식 흉내)
# ---------------------------------------------------------
def make_synthetic_scene(
    H=240, W=240,
    plate_depth_m=0.60,      # 접시 중심까지 거리
    plate_d_px=188,          
    plate_d_cm=26.0,         
    food_radius_px=60,       
    max_height_m=0.022,      
    noise_std_m=0.0015       
):
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W // 2, H // 2

    # 접시 마스크
    plate_r_px = plate_d_px / 2.0
    plate_mask = ((xx - cx)**2 + (yy - cy)**2) <= plate_r_px**2

    # 음식 마스크
    food_mask = ((xx - cx)**2 + (yy - cy)**2) <= food_radius_px**2

    # 음식 높이 (포물면)
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    h_true = np.zeros_like(r, dtype=np.float32)
    inside = r <= food_radius_px
    h_true[inside] = max_height_m * (1.0 - (r[inside] / float(food_radius_px))**2)

    # 접시 깊이 (약간 기울임 효과 추가 가능하지만 여기선 평평 가정)
    Z_plate_true = np.full((H, W), plate_depth_m, dtype=np.float32)

    # 장면 깊이 = 접시깊이 - 음식높이 (카메라에 더 가까워짐)
    Z_scene = Z_plate_true.copy()
    Z_scene[food_mask] = Z_plate_true[food_mask] - h_true[food_mask]

    # 노이즈 추가
    if noise_std_m > 0:
        noise = np.random.normal(0.0, noise_std_m, size=(H, W)).astype(np.float32)
        Z_scene = Z_scene + noise

    # 배경 NaN 처리
    Z_scene[~plate_mask] = np.nan

    meta = {"units": "meters", "shape": (H, W), "note": "Synthetic Data"}
    return Z_scene, food_mask.astype(bool), plate_mask.astype(bool), plate_d_cm, meta


# ---------------------------------------------------------
# 부피 공식 수정
# ---------------------------------------------------------
def volume_from_depth_perspective_corrected(
        Z_scene: np.ndarray,
        food_mask: np.ndarray,
        plate_mask: np.ndarray,
        plate_d_cm: float,
        density_g_per_ml: float = 1.0
    ):
    """
    1. 접시 깊이와 크기로 카메라 초점거리(Focal Length) 역추산
    2. 접시 영역을 평면 피팅(Plane Fitting)하여 기준 바닥면(Base) 생성
    3. (바닥면 - 측정깊이) * (깊이에 따른 픽셀 면적) 적분
    """
    H, W = Z_scene.shape
    
    # 1. 접시 영역 깊이값 추출
    plate_only_mask = plate_mask & (~food_mask) & (~np.isnan(Z_scene))
    if not plate_only_mask.any():
        plate_only_mask = (~np.isnan(Z_scene)) # fallback

    plate_z_vals = Z_scene[plate_only_mask]
    plate_dist_ref_m = float(np.median(plate_z_vals)) if plate_z_vals.size > 0 else 1.0

    # 2. 접시 지름(px) 및 초점거리(f_px) 추정
    #    f_px = (이미지상크기 * 거리) / 실제크기
    plate_d_px = estimate_plate_diameter_px(plate_mask)
    if plate_d_px <= 0: plate_d_px = W // 2

    plate_d_m = plate_d_cm / 100.0
    if plate_d_m > 0:
        f_px = (plate_d_px * plate_dist_ref_m) / plate_d_m
    else:
        f_px = (W + H) / 2.0  # fallback

    # 3. 접시 평면 피팅 (Ax + By + C = Z)
    #    이미지 좌표 (u, v)에서 깊이 Z를 평면으로 근사
    grid_y, grid_x = np.indices((H, W))
    
    p_ys = grid_y[plate_only_mask]
    p_xs = grid_x[plate_only_mask]
    p_zs = Z_scene[plate_only_mask]

    # 행렬 A = [x, y, 1], b = [z]
    if p_xs.size > 10:
        A_mat = np.column_stack((p_xs, p_ys, np.ones_like(p_xs)))
        # 최소자승법
        plane_coeffs, _, _, _ = np.linalg.lstsq(A_mat, p_zs, rcond=None)
    else:
        # 데이터가 너무 없으면 그냥 상수 평면 (기울기 0)
        plane_coeffs = np.array([0.0, 0.0, plate_dist_ref_m])

    # 4. 음식 영역 높이 계산
    #    높이 = (평면으로 추정한 바닥 Z) - (실제 측정된 Z)
    #    * 카메라 좌표계에서 Z값이 작을수록 가깝다. 바닥이 더 머니까 Z값이 더 큼.
    valid_food = food_mask & (~np.isnan(Z_scene))
    
    f_ys, f_xs = np.where(valid_food)
    z_measured = Z_scene[valid_food]
    
    # 해당 픽셀 위치의 "접시 바닥" 깊이 추정
    food_coords = np.column_stack((f_xs, f_ys, np.ones_like(f_xs)))
    z_base = food_coords @ plane_coeffs

    heights_m = z_base - z_measured
    heights_m = np.maximum(0.0, heights_m) # 음수(접시보다 파고든 노이즈) 제거

    # 5. 부피 적분 (원근 보정: 깊이에 따라 픽셀 면적이 다름)
    #    픽셀 하나의 실제 면적 A = (Z / f)^2
    pixel_areas_m2 = (z_measured / f_px) ** 2
    volumes_m3 = heights_m * pixel_areas_m2

    total_volume_m3 = np.sum(volumes_m3)
    total_volume_ml = total_volume_m3 * 1e6
    total_mass_g = total_volume_ml * density_g_per_ml

    # 6. 결과 딕셔너리 구성 (main함수 출력을 위해 필요한 키 포함)
    
    # 참고용: 접시 기준 깊이에서의 대략적 스케일 (cm/px)
    ref_scale_cm_per_px = (plate_dist_ref_m / f_px) * 100.0 if f_px > 0 else 0
    
    # 참고용: 음식 면적 (단순 합산)
    food_area_px = int(valid_food.sum())
    # 면적은 깊이마다 다르지만, 대략적 표시를 위해 평균 깊이 기준 계산
    avg_food_depth = np.mean(z_measured) if z_measured.size else plate_dist_ref_m
    avg_pixel_area_cm2 = ((avg_food_depth / f_px) * 100.0) ** 2
    food_area_cm2 = food_area_px * avg_pixel_area_cm2

    h_max_cm = float(np.max(heights_m) * 100.0) if heights_m.size > 0 else 0.0

    return {
        "volume_ml": total_volume_ml,
        "mass_g": total_mass_g,
        "focal_length_px": f_px,
        "plate_d_px": plate_d_px,
        "plate_dist_ref_m": plate_dist_ref_m,
        "scale_cm_per_px": ref_scale_cm_per_px, # 근사값(참고용)
        "food_area_px": food_area_px,
        "food_area_cm2": food_area_cm2,
        "food_h_max_cm": h_max_cm
    }


# ---------------------------------------------------------
# [YOLO] 추론 및 마스크 변환
# ---------------------------------------------------------
PLATE_LIKE = {'plate', 'bowl', 'cup', 'wine glass', 'tray', 'dining table'}
FOOD_LIKE  = {
    'food', 'rice', 'noodles', 'pizza', 'sandwich', 'salad', 'cake', 'donut',
    'banana','apple','orange','broccoli','carrot','hot dog','burger','steak',
    'bread'
}

def yolo_seg_to_masks(image_path: str, yolo_weights: str, depth_shape_hw):
    if not _YOLO_AVAILABLE:
        raise RuntimeError("ultralytics 미설치")
    
    H, W = depth_shape_hw
    model = YOLO(yolo_weights)
    # conf=0.25는 상황에 맞게 조절
    results = model(image_path, imgsz=640, conf=0.15, verbose=False)
    
    if not results:
        return np.zeros((H, W), bool), np.zeros((H, W), bool), []

    res = results[0]
    if res.masks is None:
        return np.zeros((H, W), bool), np.zeros((H, W), bool), []

    masks = res.masks.data.cpu().numpy() # (N, h_mask, w_mask)
    clses = res.boxes.cls.cpu().numpy().astype(int)
    names = res.names

    food_mask  = np.zeros((H, W), dtype=bool)
    plate_mask = np.zeros((H, W), dtype=bool)
    picked = []

    for mi, ci in zip(masks, clses):
        cls_name = names[ci]
        # 마스크 리사이즈 (Depth Map 크기로)
        m_resized = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        if cls_name in PLATE_LIKE:
            plate_mask |= m_resized
            picked.append(('plate', cls_name))
        elif cls_name in FOOD_LIKE:
            food_mask |= m_resized
            picked.append(('food', cls_name))
    
    return food_mask, plate_mask, picked


# ---------------------------------------------------------
# [파이프라인] 실행
# ---------------------------------------------------------
def run_pipeline(args):
    # 1. Input Load
    if args.mode == "synth":
        np.random.seed(42)
        Z_scene, food_mask, plate_mask, plate_d_cm, meta = make_synthetic_scene(
            plate_cm=args.plate_cm, 
            plate_depth_m=0.60
        )
        print(f"[Synth] 생성 완료. (접시 {args.plate_cm}cm 설정)")
        picked = [("System", "Synthetic Data")]
    else:
        # Real Data
        if not args.image or not args.depth:
            raise ValueError("--image와 --depth 경로가 필요합니다.")
        
        # Load Depth
        if args.depth.lower().endswith('.npy'):
            Z_scene = np.load(args.depth).astype(np.float32)
        else:
            Z_raw = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
            if Z_raw is None: raise FileNotFoundError(args.depth)
            Z_scene = Z_raw.astype(np.float32)
        
        Z_scene *= args.depth_scale # 단위 보정

        # YOLO Inference
        food_mask, plate_mask, picked = yolo_seg_to_masks(args.image, args.yolo_weights, Z_scene.shape)

        plate_d_cm = args.plate_cm

    # 2. 접시 마스크 안전장치
    if not plate_mask.any():
        print("[Warning] 접시(Plate)를 감지하지 못했습니다! 전체 화면을 기준으로 계산합니다.")
        plate_mask = np.ones_like(food_mask, dtype=bool)

    # 3. 계산 (Perspective + Plane Fitting)
    result = volume_from_depth_perspective_corrected(
        Z_scene=Z_scene,
        food_mask=food_mask,
        plate_mask=plate_mask,
        plate_d_cm=plate_d_cm,
        density_g_per_ml=args.density
    )
    
    return result, picked


# ---------------------------------------------------------
# [Main]
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synth", "yolo"], default="synth")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--depth", type=str, default=None)
    parser.add_argument("--plate-cm", type=float, default=26.0, help="접시 실제 지름(cm)")
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--yolo-weights", type=str, default="yolov8x-seg.pt")
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Depth value scale to Meter")

    args = parser.parse_args()

    try:
        res, picked = run_pipeline(args)
    except Exception as e:
        print(f"\n[Error] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 출력
    print("\n" + "="*40)
    print(f" 감지된 객체: {len(picked)} 개")
    for kind, name in picked:
        print(f"  - [{kind}] {name}")
    print("="*40)

    print(f"1. 접시 분석")
    print(f"  - 실제 지름 설정 : {args.plate_cm} cm")
    print(f"  - 이미지상 지름  : {res['plate_d_px']} px")
    print(f"  - 추정 초점거리  : {res['focal_length_px']:.1f} px")
    print(f"  - 접시 거리(Ref) : {res['plate_dist_ref_m']*100:.1f} cm")
    print(f"  - 기준 스케일    : 1 px ≈ {res['scale_cm_per_px']:.4f} cm (거리별 가변)")

    print("-" * 40)
    print(f"2. 음식 분석 (Perspective & Tilt Corrected)")
    print(f"  - 음식 투영 면적 : {res['food_area_px']} px (약 {res['food_area_cm2']:.1f} cm²)")
    print(f"  - 최대 높이(추정): {res['food_h_max_cm']:.2f} cm")
    print(f"  - 추정 부피      : {res['volume_ml']:.1f} ml")
    print(f"  - 추정 질량      : {res['mass_g']:.1f} g (밀도 {args.density})")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
