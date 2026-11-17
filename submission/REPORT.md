# Báo cáo Day 08 — 2A202603253 - Nguyễn Thảo Linh

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | motorbike=0.10, còn lại mặc định 0.30 | YOLO26m phát hiện tốt 6 lớp COCO cơ bản (r02, r03). motor đạt recall 90% (9/10 vật gold) ở ngưỡng 0.10; bicycle, bus, truck có mẫu < 5 vật nên giữ mặc định theo luật mẫu nhỏ; person và car không có ngưỡng <= 0.5 đạt 90% recall nên giữ 0.30 để tránh sinh quá nhiều false positive |
| three_wheeler | YOLOE-26s, prompt: tuk-tuk, cyclo, cargo tricycle | 0.15 | COCO đóng không có lớp xe ba bánh nên YOLO26 bỏ sót hoặc gọi nhầm sang car/truck. YOLOE-26s tìm theo text prompt đạt recall 100% (7/7 vật gold) và precision 78% tại ngưỡng 0.15 (r04) |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "motorbike=0.1,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: SAM 2.1 base (`sam2.1_b.pt`) vì cho chất lượng đường biên mask bám sát các chi tiết nan hoa, khung sắt và bánh xe; cân bằng tốt giữa thời gian suy luận và độ chính xác phân đoạn; không yêu cầu đăng nhập token gated như SAM 3 và vượt trội về độ chi tiết so với MobileSAM (r05).
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? Lỗi nhận nhầm loại xe đặc thù địa phương (tuk-tuk, xe ba gác bị YOLO26 gán thành `car` hoặc `truck` như trên `img_002.jpg` và `img_013.jpg`). Dù tăng conf lên 0.5 hay hạ xuống 0.1 thì YOLO26 vẫn gán sai do tập nhãn đóng. Cách sửa: Dùng mô hình open-vocabulary YOLOE-26s để bắt riêng `three_wheeler`, sau đó trong `merge_prelabels.py` đặt file YOLOE lên trước YOLO26 để box `three_wheeler` đè và loại bỏ box `car`/`truck` nhận nhầm khi IoU >= 0.6.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  - Điểm cao mà sai: Trên `img_002.jpg`, YOLO26 nhận diện chiếc tuk-tuk tiền cảnh thành `car` với điểm tin cậy rất cao `0.89` và `truck` điểm `0.73`; trên `img_013.jpg`, tuk-tuk cũng bị nhận là `car` với điểm `0.72`. Điểm tin cậy cao của máy không đảm bảo nhãn đúng ngữ nghĩa.
  - Điểm thấp mà đúng: Trên `img_004.jpg`, chiếc tuk-tuk bị che khuất một phần trong góc chỉ đạt điểm `0.34` - `0.41` từ detector nhưng thực tế là xe có thật.
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: Phép đo in-sample (tối ưu ngưỡng trên 4 ảnh gold rồi đo lại trên chính 4 ảnh đó) luôn cho kết quả đẹp hơn thực tế trên 24 ảnh batch. Mẫu 4 ảnh chỉ có 1 xe đạp, 0 xe buýt, 1 xe tải nên hoàn toàn không đủ độ tin cậy thống kê để thiết lập ngưỡng riêng cho các lớp hiếm.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 553 | 553/620 = 89.2% | 620 box máy |
| edited | 48 | 48/620 = 7.7% | 620 box máy |
| deleted | 3 | 3/620 = 0.5% | 620 box máy |
| added | 2 | 2/603 = 0.3% | 603 đối tượng sau duyệt |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? Lớp `three_wheeler` và `motorbike` ở nền xa. Với `three_wheeler`, YOLOE có xu hướng vẽ box lỏng và nhận nhầm một phần thùng xe tải (trên `img_002` và `img_013` phải chỉnh sửa lại 7/39 box = 17.9%). Với `motorbike`, do hạ ngưỡng xuống 0.10 để đạt recall 90% nên máy sinh nhiều box thừa ở nền xa phải xóa bỏ (3 box deleted) hoặc viền hở phải kéo khít lại (20 box edited).
- So với gold: precision / recall của **máy**: Precision = 40.2% (35/87) [KTC 95%: 31–51%], Recall = 85.4% (35/41) [KTC 95%: 72–93%] (mẫu số: 41 đối tượng gold trên 4 ảnh gold).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 11/12 = 91.7% [KTC 95%: 65–99%]; loại nào lọt: 1 bẫy `loose` trên `img_025.jpg` (box người đi bộ bị nới rộng phần đầu và vai); vì sao (bạn đã bỏ bước nào của checklist)? Người đi bộ ở cự ly xa, phông nền tối màu trùng với màu áo nên khi nhìn bao quát mắt dễ bị đánh lừa là box đã khớp. Người duyệt đã lướt nhanh bước 3 của checklist (phóng to soi kỹ 4 cạnh viền box).
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? Nếu accepted rất cao (>95%) mà bắt bẫy thấp (<80%), người duyệt đã rơi vào bẫy tâm lý thiên kiến tự động hóa (automation bias), bấm "Accept" theo quán tính mà không thực sự kiểm tra. Ngược lại, bắt bẫy cao (91.7%) và bắt được trọn vẹn 3/3 lỗi xóa mất (`missing`) và 3/3 lỗi đổi nhãn (`wrong_class`) chứng minh người duyệt đã áp dụng nghiêm túc quy trình quét độc lập toàn cảnh trước khi đối chiếu gợi ý của máy.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 6.0 phút/ảnh (24 phút / 4 ảnh) vs duyệt 2.8 phút/ảnh (68 phút / 24 ảnh). Tốc độ tăng hơn gấp đôi (~2.14 lần). Thiên lệch (bạn gán gold trước nên đã quen cảnh): Do đã gán 4 ảnh gold trước nên mắt đã quen với đường phố châu Á, đặc điểm xe máy đông đúc và các góc chụp xe tuk-tuk, nên một phần tốc độ tăng lên ở lô batch đến từ hiệu ứng quen tay chứ không hoàn toàn do AI.
- Kết luận: Máy RẤT ĐÁNG DÙNG, đặc biệt cho các lớp phổ biến (`person`, `motorbike`, `car`) vì giúp tiết kiệm hơn 50% thời gian vẽ tay ban đầu. Tuy nhiên, AI KHÔNG THỂ thay thế hoàn toàn con người trong các tình huống: lớp địa phương ngoài COCO (`three_wheeler`), vật nhỏ ở cự ly xa (<20px) cần gom vùng `ignore_region`, và các cụm xe máy chen chúc dày đặc.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. **Quy tắc Ẩn nhãn trước (Blind First Pass):** Trước khi nhìn nhãn máy, luôn ấn nút ẩn toàn bộ bounding box, dùng mắt quét toàn ảnh từ trái qua phải trong 15 giây và đếm nhẩm số lượng người/xe để tự tạo danh sách kiểm chứng độc lập.
  2. **Quy tắc Zoom 100% soi 4 cạnh (Boundary Inspection):** Với mỗi box được máy gợi ý, bắt buộc phóng to tối thiểu 100% và kiểm tra 4 cạnh (đặc biệt cạnh trên đầu người và cạnh dưới bánh xe) để phát hiện lỗi box lỏng (`loose`) hoặc box ôm cả nền phía sau.
  3. **Quy tắc Đếm bẫy kiểm soát (Trap Catch Rate >= 80%):** Luôn duy trì bộ ảnh bẫy ẩn định kỳ; nếu tỷ lệ bắt bẫy giảm xuống dưới 80%, người duyệt phải dừng ngay lập tức, nghỉ giải lao 10 phút để phục hồi sự tập trung và rà soát lại toàn bộ lô nhãn gần nhất.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013.jpg | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 pos, 1 neg) | Không gồm người lái (điểm neg `720,300` đã tách rời người lái); ôm trọn mui và thân xe | Điểm dương đơn lẻ ban đầu chỉ bắt phần huy hiệu thân xe; cần click dương thứ hai để kéo mở rộng lên mái che; viền bánh trước hơi dính vệt bóng mặt đường | `tuktuk_img013_sam2.jpg` |
| img_013.jpg | MobileSAM (`mobile_sam.pt`) | 3 (2 pos, 1 neg) | Tách được người lái nhưng mask bị vỡ thành 2 polygon rời rạc (thiếu mui sau) | Đặc trưng nông hơn SAM 2.1 làm mask bị rỗ ở các thanh chống mảnh; tổng diện tích chỉ đạt 35.2% ảnh so với 45.4% của SAM 2.1 | `tuktuk_img013_sam1.jpg` |
| img_004.jpg | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 pos, 1 neg) | Không gồm người đi bộ đứng sát cạnh; ôm khít thân xe tuk-tuk bị che khuất | Xe bị ô tô bên cạnh che khuất phần đầu, viền mask bám rất sát mép cản nhưng mép kính chắn gió hơi lẹm do chói sáng | `tuktuk_img004_sam2.jpg` |
| img_017.jpg | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 pos, 1 neg) | Điểm neg `645,355` đã loại trừ trọn vẹn người lái bên trong cabin | Khung xe tuk-tuk dạng ống sắt rỗng khiến SAM dễ nuốt trọn cả nền nhà phía sau qua khung cửa sổ; cần điểm âm để chặn biên tiếp xúc | `tuktuk_img017_sam2.jpg` |

- Đối chiếu guideline mục 4.2: Quy tắc 4.2 quy định người lái là `person` riêng, phương tiện chỉ bao gồm khung, mui và bánh xe. Quy tắc này cực kỳ hữu ích làm căn cứ đặt điểm loại trừ `--neg` lên thân người lái. Tuy nhiên, guideline v1.0 thiếu quy định chi tiết về việc có bắt buộc khoét rỗng vùng nền nhìn xuyên qua ô thoáng cửa xe hay được phép phủ kín toàn khối thân xe (cần đưa vào Calibration ở Part 6).
- *(Tuỳ chọn)* Interactor SAM trong CVAT: Interactor trên CVAT cho phép click trực tiếp trên canvas (chuột trái = điểm dương, chuột phải = điểm âm) và nhìn thấy mask co giãn trực tiếp tức thì, dễ tinh chỉnh hơn việc mò tọa độ pixel bằng script CLI. Tuy nhiên, với máy chạy CPU, việc chạy Nuclio/SAM trong CVAT tốn nhiều RAM và độ trễ phản hồi mỗi click khoảng 3-5 giây, trong khi script CLI chạy offline ổn định và dễ kiểm soát tham số reproducibility.

## 6. Calibration (Part 6)

- Số quyết định: 4 quyết định (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): `CAL-01` (khắc phục lỗi máy closed-set YOLO26 nhận nhầm tuk-tuk thành `car`/`truck` bằng việc ưu tiên YOLOE đè lên YOLO26 và khẳng định quy tắc mọi xe ba bánh có mui đều là `three_wheeler`) và `CAL-02` (chốt câu hỏi mở số 1: phân loại xe đạp điện có bàn đạp trợ lực là `bicycle` và xe máy điện scooter không bàn đạp là `motorbike`).
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: Đã bổ sung 4 quy tắc: phân định xe điện theo bàn đạp cơ học (mục 2.2); quy chuẩn ranh giới box và mask cho khoảng hở cửa sổ xe ba bánh (mục 4.2); khẳng định phân loại `three_wheeler` cho tuk-tuk bất kể kích thước (mục 2 & 4.3); và bổ sung quy tắc phóng to 100% kiểm tra biên các vật ở xa chống thiên kiến tự động hóa (mục 4.1.7).
- Nguyên nhân gốc: Phân hóa rõ ràng giữa: (1) Guideline v1.0 ban đầu còn câu hỏi mở chưa có quy định (xe điện, ô thoáng cửa sổ); (2) Giới hạn cấu trúc của mô hình closed-set (YOLO26 chỉ có 80 lớp COCO chuẩn nên bắt buộc gán nhầm tuk-tuk sang xe khác); và (3) Thiên kiến tự động hóa của con người khi rà soát nhanh các box lỏng ở hậu cảnh tương phản thấp.

## 7. QC và quyết định (Part 6)

- Mẫu: 5 ảnh (0 ảnh `can_review` + 5 ngẫu nhiên, seed 42); người kiểm: Nguyễn Thảo Linh (**hạn chế**: người duyệt cũng chính là người kiểm tra QC, tạo nguy cơ thiên lệch xác nhận - confirmation bias).
- Lỗi theo **đối tượng**: 0/126 = 0.0% · theo **ảnh**: 0/5 = 0.0% (ngưỡng Release ≤ 2.0%).
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: Không ghi nhận lỗi nào trong mẫu 5 ảnh ngẫu nhiên được kiểm định (`img_003.jpg`, `img_006.jpg`, `img_014.jpg`, `img_031.jpg`, `img_035.jpg`).
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): Cận trên khoảng tin cậy Wilson 95% của tỷ lệ ảnh lỗi thực tế lên tới **43%**. Với cỡ mẫu chỉ 5 ảnh ngẫu nhiên, về mặt thống kê toán học chưa thể chứng minh chắc chắn tỷ lệ lỗi toàn quần thể $\le 2\%$; cần tiếp tục kiểm tra định kỳ trong quá trình vận hành tiếp theo.
- Script kiểm được / không kiểm được: Script kiểm tra được cú pháp schema COCO, đối chiếu IoU toán học, kiểm tra box âm/quá nhỏ; nhưng script KHÔNG THỂ kiểm tra được độ khít thực tế của viền theo ngữ cảnh thị giác người nhìn, việc gom vùng `ignore_region` có thỏa đáng không, hoặc ca cả máy lẫn người cùng bỏ sót một vật thể ẩn sâu trong bóng râm.

**Quyết định: RELEASE**, ngày 2026-11-17. Lý do (đi theo Hold → Rework → Release):
1. Kiểm tra điều kiện HOLD: Không còn câu hỏi mở hay khoảng trống guideline (toàn bộ đã chốt trong v1.1), không có ảnh dữ liệu bị hỏng ➔ Không HOLD.
2. Kiểm tra điều kiện REWORK: Tỷ lệ lỗi theo ảnh (0.0%) và theo đối tượng (0.0%) đều $\le 2\%$, tỷ lệ bắt bẫy đạt 91.7% ($\ge 80\%$), không có lỗi hệ thống lặp lại ➔ Không REWORK.
3. Thỏa mãn toàn diện các tiêu chí nghiệm thu ➔ Chốt quyết định bàn giao **RELEASE**.
Nếu Rework / Hold, việc cần làm tiếp: Không áp dụng vì bộ dữ liệu đã đạt chuẩn Release.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): Sử dụng Antigravity AI Assistant để hỗ trợ thực thi pipeline, chạy các script CLI, gộp nhãn và định dạng báo cáo. Toàn bộ số liệu và kết quả đều được trích xuất và đối chiếu trực tiếp từ các file kết quả thực tế trên đĩa (`review_stats.json`, `qc_report_auto.md`, `model_log.csv`).
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? HOÀN TOÀN KHÔNG. 4 ảnh gold (`gold_manual.json`) được gán tay độc lập từ đầu ở Part 1 trước khi chạy bất kỳ mô hình pre-label nào.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Windows 11, CPU (Intel Core), Python 3.12 trong môi trường ảo `.venv`, CVAT local.
- Dùng cache `data/cache/` thay vì tự chạy? KHÔNG dùng cache. Tự chạy trực tiếp toàn bộ các mô hình YOLO26m, YOLOE-26s, SAM 2.1 base và MobileSAM trên máy local.
- Lỗi / bug script đã gặp (nếu có): Cần bọc ngoặc kép quanh các chuỗi tham số có dấu phẩy trên PowerShell để tránh bị cắt chuỗi; cần chỉ định `YOLO_CONFIG_DIR` trỏ vào thư mục workspace cục bộ để tránh lỗi quyền ghi file config mặc định của Ultralytics trong môi trường sandbox.
- Một điều bạn làm khác lần sau: Sẽ phóng to 100% và rà soát kỹ lưỡng hơn các đối tượng nhỏ ở nền xa ngay từ lượt quét đầu tiên để đạt mục tiêu bắt trọn vẹn 12/12 bẫy nhân tạo.
