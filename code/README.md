# Draft_v2 — mã nguồn của đề án (PWL-SVM)

Python 3.12, NumPy 1.26, SciPy 1.13, scikit-learn 1.5, matplotlib 3.9 (font TeX Gyre Termes của TeX Live).
Dữ liệu UCI được tải qua OpenML (`fetch_openml`, cần mạng ở lần chạy đầu). Mọi hạt giống ngẫu nhiên đều cố định.
Hình được ghi vào `../de_an/figures/`, bảng và đoạn văn có số liệu vào `../de_an/chapters/`, kết quả thô vào `results/`.

## Mô-đun

- `pwlsvm.py` — ánh xạ đặc trưng cộng tính / HH / GHH (`PWLFeatures`, gồm hai tùy chọn của đề án: nút theo phân vị và
  chuẩn hóa `‖p‖₂ = 1`) và mô hình `PWLSVM` (PWL-C-SVM bằng LIBSVM hoặc LIBLINEAR, PWL-LS-SVM bằng hồi quy ridge),
  theo Huang, Mehrkanoon & Suykens (2013).
- `pwlsvm_monotone.py` — PWL-LS-SVM cộng tính có ràng buộc đơn điệu (tham số hóa theo hệ số góc, giải bằng BVLS), Mục 3.5.5.
- `scorecard.py` — thẻ điểm đối chứng: hồi quy logistic trên các biến đã rời rạc hóa.
- `baselines.py` — LS-SVM hạt nhân Gauss, SVM hạt nhân giao. `datasets2d.py`, `datasets_uci.py`, `credit_data.py` — dữ liệu.
- `plotstyle.py` — kiểu hình chung (dấu phẩy thập phân, bảng màu).

## Tái lập hình và bảng (theo thứ tự)

| Lệnh | Sinh ra | Thời gian (1 máy, 11 lõi) |
|---|---|---|
| `python3 fig_theory_ch1.py && python3 fig_theory_ch2.py && python3 fig_theory.py` | Hình 1.1–1.5, 2.1–2.4, 2.6–2.8 (hình minh họa lý thuyết; Hình 2.5 vẽ bằng TikZ trong LaTeX) | < 1 phút |
| `python3 verify.py` | kiểm chứng gốc = đối ngẫu (Mục 3.1), `results/verify.json` | < 1 phút |
| `python3 exp_2d.py` | Hình 3.1, Bảng 3.1 (dữ liệu hai chiều) | ≈ 7 phút |
| `python3 exp_cosexp.py` | Hình 3.2 (Cosexp) | ≈ 15 phút |
| `PYTHONWARNINGS=ignore python3 exp_benchmark.py` | `results/benchmark.json` (mười bộ dữ liệu, Mục 3.3.2) | vài giờ (Spambase chiếm phần lớn) |
| `python3 exp_spam_hhnorm.py` | `results/spam_hhnorm.json` (HH với `‖p‖₂ = 1` trên Spambase) | ≈ 15 phút |
| `python3 exp_sensitivity.py` | `results/sensitivity.json` (Mục 3.3.3) | ≈ 30 phút |
| `PYTHONWARNINGS=ignore python3 exp_credit.py` | `results/credit.json`, `results/credit_interpret.json` (Mục 3.5) | ≈ 1 giờ |
| `python3 exp_credit_scorecard.py` | `results/credit_scorecard.json` (thẻ điểm đối chứng) | ≈ 1 phút |
| `python3 credit_stability.py` | `results/credit_shapes_all.json` (độ ổn định của các hàm đóng góp) | < 1 phút |
| `python3 exp_monotone.py` | `results/credit_monotone.json` (ràng buộc đơn điệu, Mục 3.5.5) | ≈ 1 phút |
| `PYTHONWARNINGS=ignore python3 exp_cost.py` | `results/cost.json` (Mục 3.4, đo trên một luồng) | ≈ 20 phút |
| `python3 make_tables.py` | các tệp `tab_*.tex` của Chương 3 và Phụ lục B (Bảng 3.3–3.5, 3.7, PL.1) | giây |
| `python3 fig_results.py` | Hình 3.3–3.10 | giây |
| `python3 credit_hgb_trees.py` | `results/credit_hgb_trees.json` (số cây của tổ hợp cây tăng cường, dùng trong văn bản) | < 1 phút |
| `python3 fill_numbers.py` | `chuong3_bench_b.tex`, `chuong3_cost.tex`, `chuong3_credit_b.tex`, `chuong3_end.tex`, `ketluan.tex` từ các tệp `.tpl` cùng tên | giây |

`run_spam.sh`, `run_rest.sh` và `run_extra.sh` là các kịch bản đã dùng để chạy các thí nghiệm dài ở chế độ nền; `exp_cost.py` chờ hai tệp cờ `results_spam.done` và `results_extra.done` để chỉ đo thời gian khi máy rảnh.
Thời gian đo trong `exp_cost.py` chỉ có ý nghĩa khi không có tiến trình nặng nào khác chạy song song.

## Quy ước về số liệu trong văn bản

Mọi con số của Mục 3.3.2–3.6 và phần Kết luận được lấy từ `results/*.json` bởi `fill_numbers.py`; muốn sửa câu chữ,
hãy sửa tệp `.tpl` rồi chạy lại script, không sửa trực tiếp tệp `.tex` tương ứng. Script dừng lại với thông báo lỗi nếu một
nhận định định tính của văn bản (chẳng hạn “khác biệt có ý nghĩa thống kê”) không còn đúng với kết quả mới.
Các đoạn diễn giải hình dạng các hàm đóng góp được viết tay trong `../de_an/chapters/shapes_text.tpl`, `stab_text.tpl` và `explain_text.tpl`; chúng được chèn vào `chuong3_credit_b.tex` cùng với các con số.

## Tệp từ bản mẫu (Draft_v1), không dùng cho bản đầy đủ

`exp_uci_preview.py`, `make_uci_table.py`, `probe_credit.py` — thí nghiệm sơ bộ của bản mẫu, giữ lại để đối chiếu.
