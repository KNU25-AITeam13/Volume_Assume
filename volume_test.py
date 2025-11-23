# -*- coding: utf-8 -*-
"""
.\.venv\Scripts\Activate.ps1

(접시 지름 스케일 + 상수 픽셀면적)로
Depth Pro 깊이맵    + (YOLOv8-seg로 만든) 음식/접시 마스크를 이용하여   
부피(ml)와 질량(g)을 계산하는 단일 스크립트.

- YOLO가 없어도 합성 데이터를 생성하여 즉시 테스트 가능.
- 실제 적용 시에는 --mode yolo 로 이미지/깊이/가중치를 입력.


실행 예시:
  데모:
    python volume_test.py

  YOLO + 실제:
    python .\volume_test.py --mode yolo --image .\scene.jpg --depth .\depth.npy --plate-cm 26 --yolo-weights yolov8x-seg.pt

"""

import argparse
import sys
import numpy as np
import cv2

# YOLO 사용 여부를 동적으로 결정
try:
    from ultralytics import YOLO  # pip install ultralytics
    _YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    _YOLO_AVAILABLE = False


# ---------------------------------------------------------
# [유틸] 접시 지름(픽셀) 추정: 행 단위 폭의 최대값을 지름으로 사용
#  - 타원 피팅 대신 의존성 없이 간단히 최대 폭을 지름으로
# ---------------------------------------------------------
def estimate_plate_diameter_px(plate_mask: np.ndarray) -> int:
    H, W = plate_mask.shape
    max_width = 0
    for y in range(H):
        xs = np.where(plate_mask[y])[0]
        if xs.size:
            width = xs.max() - xs.min() + 1
            if width > max_width:
                max_width = width
    return int(max_width)


# ---------------------------------------------------------
# [유틸] 합성 데이터 생성 (Depth Pro 형식 흉내)
#  - Z_scene: H×W float32 (단위 m), 음식 있는 곳은 접시보다 가까움
#  - food_mask / plate_mask: 불리언 마스크
# ---------------------------------------------------------
def make_synthetic_scene(
    H=240, W=240,
    plate_depth_m=0.60,      # 접시(테이블)까지 거리(깊이, m)
    plate_d_px=188,          # 접시 지름(픽셀) - 합성 장면에서의 값
    plate_d_cm=26.0,         # 접시 실제 지름(cm) - 스케일에 사용
    food_radius_px=60,       # 음식 바닥 반경(픽셀)
    max_height_m=0.022,      # 음식 최대 높이(약 2.2 cm)
    noise_std_m=0.0015       # 깊이 노이즈(표준편차, m)
):
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W // 2, H // 2

    # 접시 마스크(원형)
    plate_r_px = plate_d_px / 2.0
    plate_mask = ((xx - cx)**2 + (yy - cy)**2) <= plate_r_px**2

    # 음식 마스크(원형)
    food_mask = ((xx - cx)**2 + (yy - cy)**2) <= food_radius_px**2

    # 음식 높이맵(포물면 모양)
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    h = np.zeros_like(r, dtype=np.float32)
    inside = r <= food_radius_px
    h[inside] = max_height_m * (1.0 - (r[inside] / float(food_radius_px))**2)

    # 접시 깊이(평평)
    Z_plate_true = np.full((H, W), plate_depth_m, dtype=np.float32)

    # 장면 깊이: 음식 있는 곳은 접시보다 높이 만큼 더 가깝게
    Z_scene = Z_plate_true.copy()
    Z_scene[food_mask] = Z_plate_true[food_mask] - h[food_mask]

    # 노이즈 추가 (Depth Pro 잔오차 흉내)
    if noise_std_m > 0:
        noise = np.random.normal(0.0, noise_std_m, size=(H, W)).astype(np.float32)
        Z_scene = Z_scene + noise

    # 접시 바깥은 관측 제외로 NaN
    Z_scene[~plate_mask] = np.nan

    meta = {"units": "meters", "shape": (H, W), "note": "synthetic DepthPro-like depth map"}
    return Z_scene, food_mask.astype(bool), plate_mask.astype(bool), plate_d_cm, meta



#  입력:
#    - Z_scene    : Depth Pro 깊이맵(H×W, m)
#    - food_mask  : 음식 마스크(H×W, bool)
#    - plate_mask : 접시 마스크(H×W, bool)
#    - plate_d_cm : 접시 실제 지름(cm)
#    - density_g_per_ml : 밀도(g/ml), 기본 1.0 (물과 유사)
#  출력: dict (부피/질량 및 중간 정보)

def volume_from_depth_method_A(Z_scene: np.ndarray,
                               food_mask: np.ndarray,
                               plate_mask: np.ndarray,
                               plate_d_cm: float,
                                  density_g_per_ml: float = 1.0):
    H, W = Z_scene.shape

    # 1) 접시 지름(픽셀) → 스케일(cm/px)
    plate_d_px = estimate_plate_diameter_px(plate_mask)
    if plate_d_px <= 0:
        # 접시 마스크가 없거나 잘못되었을 때의 안전장치
        plate_d_px = max(H, W) // 2
    s_cm_per_px = plate_d_cm / float(plate_d_px)      # cm/px
    per_pixel_area_m2 = (s_cm_per_px ** 2) * 1e-4     # (cm^2 → m^2)

    # 2) 접시 기준 깊이(중앙값) → 높이맵 h = max(0, Z_plate_ref - Z_scene)
    plate_only = plate_mask & (~food_mask)
    plate_vals = Z_scene[plate_only]
    plate_vals = plate_vals[~np.isnan(plate_vals)]
    if plate_vals.size == 0:
        plate_vals = Z_scene[~np.isnan(Z_scene)]
    Z_plate_ref = float(np.median(plate_vals)) if plate_vals.size else float(np.nanmedian(Z_scene))

    h = np.zeros_like(Z_scene, dtype=np.float32)
    valid_food = food_mask & (~np.isnan(Z_scene))
    h[valid_food] = np.maximum(0.0, Z_plate_ref - Z_scene[valid_food])  # m

    # 3) 부피 적분: Σ(h) * (픽셀당 면적)
    V_m3 = float(np.nansum(h)) * per_pixel_area_m2
    V_ml = V_m3 * 1e6
    grams = V_ml * density_g_per_ml

    # 4) 보조 정보
    area_px = int(food_mask.sum())
    area_cm2 = area_px * (s_cm_per_px ** 2)
    h_max_cm = float(np.nanmax(h) * 100.0) if np.isfinite(np.nanmax(h)) else 0.0

    return {
        "volume_m3": V_m3,
        "volume_ml": V_ml,
        "mass_g": grams,
        "pixel_area_const_m2": per_pixel_area_m2,
        "scale_cm_per_px": s_cm_per_px,
        "plate_d_px": plate_d_px,
        "plate_depth_ref_m": Z_plate_ref,
        "food_area_px": area_px,
        "food_area_cm2": area_cm2,
        "food_h_max_cm": h_max_cm
    }


# ---------------------------------------------------------
# [YOLO] 이미지 → 음식/접시 마스크 만들기 (YOLOv8-seg)
#  - COCO 가중치만으로는 'plate' 클래스가 빈약할 수 있음 → 커스텀 권장
#  - 결과는 깊이맵과 동일 해상도로 NEAREST 리사이즈
# ---------------------------------------------------------
PLATE_LIKE = {'plate', 'bowl', 'cup', 'wine glass', 'tray'}
FOOD_LIKE  = {
    'food', 'rice', 'noodles', 'pizza', 'sandwich', 'salad', 'cake', 'donut',
    'banana','apple','orange','broccoli','carrot','hot dog','burger','steak'
}

def yolo_seg_to_masks(image_path: str, yolo_weights: str, depth_shape_hw):
    if not _YOLO_AVAILABLE:
        raise RuntimeError("ultralytics(YOLO)가 설치되어 있지 않습니다. pip install ultralytics")
    H, W = depth_shape_hw
    model = YOLO(yolo_weights)
    res = model(image_path, imgsz=640, conf=0.25, verbose=False)[0]

    if (res.masks is None) or (res.boxes is None) or (len(res.masks.data) == 0):
        return np.zeros((H, W), bool), np.zeros((H, W), bool), []

    masks = res.masks.data.cpu().numpy().astype(np.uint8)   # (N,h,w), 0/1
    clses = res.boxes.cls.cpu().numpy().astype(int).tolist()
    # 결과 객체/모델 어디에나 names가 있을 수 있음
    names = getattr(res, "names", None) or getattr(model.model, "names", {})

    food_mask  = np.zeros((H, W), dtype=bool)
    plate_mask = np.zeros((H, W), dtype=bool)
    picked = []

    for mi, ci in zip(masks, clses):
        cls = names.get(ci, str(ci))
        m = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        if cls in PLATE_LIKE:
            plate_mask |= m
            picked.append(('plate_like', cls))
        elif cls in FOOD_LIKE:
            food_mask |= m
            picked.append(('food_like', cls))
        else:
            continue

    return food_mask, plate_mask, picked


# ---------------------------------------------------------
# [파이프라인] YOLO + Depth Pro 결과로 부피 계산
# ---------------------------------------------------------
def run_with_yolo_and_depth(image_path: str,
                            yolo_weights: str,
                            depth_path: str,
                            plate_cm: float,
                            density: float = 1.0,
                            depth_unit_scale: float = 1.0):
    # 1) 깊이맵 로드 (npy/PNG 등) → 넘파이(m 단위)로
    if depth_path.lower().endswith(".npy"):
        Z_scene = np.load(depth_path).astype(np.float32)
    else:
        # 16비트 PNG/EXR 등 사용하는 경우를 대비한 예시
        Z_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if Z_raw is None:
            raise FileNotFoundError(f"깊이 파일을 읽을 수 없음: {depth_path}")
        Z_scene = Z_raw.astype(np.float32)
    Z_scene *= float(depth_unit_scale)  # mm→m 변환 등 필요 시 적용

    # 2) YOLO로 마스크 생성
    food_mask, plate_mask, picked = yolo_seg_to_masks(image_path, yolo_weights, Z_scene.shape)

    # 3) 접시 바깥은 관측 제외(NaN)
    if plate_mask.any():
        Z_scene = Z_scene.copy()
        Z_scene[~plate_mask] = np.nan

    # 4) 방법 A로 부피 계산
    result = volume_from_depth_method_A(
        Z_scene=Z_scene,
        food_mask=food_mask,
        plate_mask=plate_mask if plate_mask.any() else np.ones_like(food_mask, bool),
        plate_d_cm=plate_cm,
        density_g_per_ml=density
    )
    return result, picked


# ---------------------------------------------------------
# [메인] CLI
# ---------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Depth Pro + 방법A로 음식 부피 계산")
    p.add_argument("--mode", choices=["synth", "yolo"], default="synth",
                   help="synth: 합성 데모 / yolo: YOLO+실제 이미지/깊이")
    p.add_argument("--image", type=str, default=None, help="원본 이미지 경로( yolo 모드 )")
    p.add_argument("--depth", type=str, default=None, help="Depth Pro 깊이맵 경로(npy/png/exr)")
    p.add_argument("--plate-cm", type=float, default=26.0, help="접시 실제 지름(cm)")
    p.add_argument("--yolo-weights", type=str, default="yolov8x-seg.pt", help="YOLOv8-seg 가중치 경로")
    p.add_argument("--density", type=float, default=1.0, help="밀도(g/ml), 기본 1.0")
    p.add_argument("--depth-scale", type=float, default=1.0,
                   help="깊이 단위 스케일(예: mm파일이면 0.001)")

    args = p.parse_args()

    if args.mode == "synth":
        # 합성 데이터 생성 → 방법A로 계산
        np.random.seed(42)
        Z_scene, food_mask, plate_mask, plate_d_cm, meta = make_synthetic_scene(
            H=240, W=240, plate_depth_m=0.60, plate_d_px=188,
            plate_d_cm=args.plate_cm, food_radius_px=60,
            max_height_m=0.022, noise_std_m=0.0015
        )
        result = volume_from_depth_method_A(
            Z_scene=Z_scene,
            food_mask=food_mask,
            plate_mask=plate_mask,
            plate_d_cm=plate_d_cm,
            density_g_per_ml=args.density
        )
        print("=== 합성/Depth Pro 형식 메타 ===")
        print(f" - 단위: {meta['units']}, 크기: {meta['shape']}, 메모: {meta['note']}")
        print("\n=== 입력 요약 ===")
        print(f" - 접시 실제 지름: {args.plate_cm:.1f} cm")
        print(f" - 접시 지름(추정, 픽셀): {result['plate_d_px']} px")
        print(f" - 스케일: {result['scale_cm_per_px']:.4f} cm/px "
              f"(px당 면적: {(result['scale_cm_per_px']**2):.4f} cm^2/px)")
        print(f" - 음식 면적: {result['food_area_px']} px  ≈ {result['food_area_cm2']:.1f} cm^2")
        print(f" - 접시 기준 깊이(중앙값): {result['plate_depth_ref_m']*100:.2f} cm")
        print(f" - 음식 최대 높이(추정): {result['food_h_max_cm']:.2f} cm")
        print("\n=== 결과 ===")
        print(f" - 부피 ≈ {result['volume_ml']:.1f} ml")
        print(f" - 질량(ρ={args.density:.2f} g/ml 가정) ≈ {result['mass_g']:.1f} g")

    else:
        # YOLO + 실제 입력
        if not _YOLO_AVAILABLE:
            print("오류: ultralytics 가 설치되어 있지 않습니다.  pip install ultralytics", file=sys.stderr)
            sys.exit(1)
        if not args.image or not args.depth:
            print("오류: --image 와 --depth 를 모두 지정해야 합니다.", file=sys.stderr)
            sys.exit(1)

        result, picked = run_with_yolo_and_depth(
            image_path=args.image,
            yolo_weights=args.yolo_weights,
            depth_path=args.depth,
            plate_cm=args.plate_cm,
            density=args.density,
            depth_unit_scale=args.depth_scale
        )
        print("=== YOLO가 잡은 클래스(요약) ===")
        for kind, cls in picked:
            print(f" - {kind}: {cls}")
        print("\n=== 결과 ===")
        print(f" - 접시 지름(추정, 픽셀): {result['plate_d_px']} px")
        print(f" - 스케일: {result['scale_cm_per_px']:.4f} cm/px "
              f"(px당 면적: {(result['scale_cm_per_px']**2):.4f} cm^2/px)")
        print(f" - 음식 면적: {result['food_area_px']} px  ≈ {result['food_area_cm2']:.1f} cm^2")
        print(f" - 부피 ≈ {result['volume_ml']:.1f} ml")
        print(f" - 질량(ρ={args.density:.2f} g/ml 가정) ≈ {result['mass_g']:.1f} g")


if __name__ == "__main__":
    main()
