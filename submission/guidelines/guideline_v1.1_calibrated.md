# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603253 - Nguyễn Thảo Linh

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | Phân loại dựa trên bàn đạp cơ học: Nếu phương tiện có bàn đạp trợ lực (pedals) thì xếp vào `bicycle`; nếu là xe máy điện kiểu dáng scooter (không có bàn đạp cơ) thì xếp vào `motorbike` | `img_011.jpg` | CAL-02 |
| 5 | Kính chắn gió / cửa sổ hở của xe ba bánh: thuộc mask xe hay không? | Với Bounding box: bao trọn khung ngoại tiếp gồm cả phần không gian cửa sổ nhìn xuyên thấu; Với Mask phân đoạn (SAM): cho phép phủ liền khối thân xe nếu diện tích ô hở cửa sổ chiếm < 15% diện tích chiếu của xe | `img_017.jpg` | CAL-03 |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_002.jpg` | Xe tuk-tuk chở khách tiền cảnh có kích thước lớn và kính chắn gió phía trước | YOLO26m nhận nhầm thành `car` (0.89) hoặc `truck` (0.73) do tập COCO đóng | Bắt buộc gán `three_wheeler`, loại bỏ hoàn toàn box `car`/`truck` chồng lấn | Mục 2 & 4.3: Xe ba bánh có mui (tuk-tuk, xe lôi, xích lô máy) luôn là `three_wheeler`, không gán `car`/`truck` dù thân xe lớn (CAL-01) |
| `img_025.jpg` | Người đi bộ ở cự ly xa (>30m), màu quần áo tối trùng màu nền phía sau | Bẫy nhân tạo làm box bị lỏng (`loose`), người duyệt lướt qua dễ bị thiên kiến tự động hóa đánh lừa | Kéo khít lại viền box sát đỉnh đầu và gót chân người | Mục 4.1.7: Quy tắc phóng to 100% khi rà soát 4 cạnh biên đối với các đối tượng cự ly xa có độ tương phản thấp (CAL-04) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên viết sẵn | Ban hành khung hợp đồng nhãn ban đầu |
| v1.1 | 1. Phân định rõ xe đạp điện có bàn đạp (`bicycle`) vs xe máy điện scooter (`motorbike`).<br>2. Quy chuẩn hóa ranh giới box và mask cho khoảng hở cửa sổ xe ba bánh.<br>3. Khẳng định quy tắc phân loại `three_wheeler` cho tuk-tuk bất kể kích thước để chặn lỗi closed-set detector.<br>4. Thêm quy tắc kiểm soát biên box cự ly xa chống thiên kiến tự động hóa. | Chốt câu hỏi mở số 1 & 5 từ v1.0; khắc phục lỗi máy lặp YOLO26 và bẫy lọt trên `img_002`, `img_011`, `img_017`, `img_025` (CAL-01, CAL-02, CAL-03, CAL-04) |
