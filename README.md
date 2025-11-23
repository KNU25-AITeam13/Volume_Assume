# Volume_Assume
Depthpro에서 반환하는 depth.npy를 바탕으로 부피를 구하는 코드. (yolo8-seg 기준)

make_fake_depth.py는 데모 테스트용이라 무시해도 됨.

volume_test.py에서
터미널 열고

.\.venv\Scripts\Activate.ps1  //가상 환경 열기
python .\volume_test.py --mode yolo --image .\scene.jpg --depth .\depth.npy --plate-cm 26 --yolo-weights yolov8x-seg.pt  //접시:26cm 기준, 밀도 물 기준. -> 추후 수정해야 함

입력하면 부피, 질량이 나옴. (음식 사진은 scene.jpg, 깊이맵은 depth.npy) 
