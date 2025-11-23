# make_fake_depth.py
import argparse, numpy as np, cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="원본 이미지 경로 (scene.jpg)")
    ap.add_argument("--out", required=True, help="저장할 깊이맵 경로 (depth.npy)")
    ap.add_argument("--plate-depth-m", type=float, default=0.60, help="접시까지 깊이(m)")
    ap.add_argument("--bump-cm", type=float, default=2.0, help="중앙 봉우리 최대 높이(cm)")
    ap.add_argument("--radius-percent", type=float, default=0.25, help="봉우리 반경(이미지 짧은 변의 비율)")
    ap.add_argument("--noise-mm", type=float, default=1.5, help="깊이 노이즈 표준편차(mm)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {args.image}")
    H, W = img.shape[:2]

    # 기본 접시 깊이(평면)
    Z = np.full((H, W), args.plate_depth_m, dtype=np.float32)

    # 중앙 봉우리(음식 높이) 만들기: 포물면 h = hmax * (1 - (r/R)^2)
    hmax = args.bump_cm / 100.0  # cm -> m
    R = int(min(H, W) * args.radius_percent)
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W // 2, H // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    mask = r <= R
    h = np.zeros_like(Z)
    h[mask] = hmax * (1.0 - (r[mask] / float(R))**2)

    # 음식이 있을수록 카메라에 더 가까워지므로 깊이를 줄임
    Z_food = Z - h

    # 약간의 잡음 추가(Depth Pro 느낌)
    if args.noise_mm > 0:
        noise = np.random.normal(0.0, args.noise_mm/1000.0, size=Z.shape).astype(np.float32)
        Z_food = Z_food + noise

    # 저장 (미터 단위 H×W float32)
    np.save(args.out, Z_food.astype(np.float32))
    print(f"[OK] 가짜 깊이 저장: {args.out} (shape={Z_food.shape}, unit=m)")

if __name__ == "__main__":
    main()
