% ----------------------------------------------------------------------------
\section{Chi phí tính toán và bộ nhớ}\label{sec:cost}

Mục~\ref{sec:pwlsvm} dự báo rằng lợi thế của PWL-SVM nằm ở khâu dự đoán: mô hình chỉ gồm $w$, $w_0$ và tham số của ánh xạ, và việc dự đoán chỉ gồm các phép so sánh, nhân, cộng và lấy cực đại (\cref{tab:cost}). Mục này kiểm tra dự báo đó trên hai bộ dữ liệu lớn hơn: Magic với $9.510$ mẫu huấn luyện (chia đôi toàn bộ dữ liệu) và dữ liệu tín dụng sẽ được mô tả ở Mục~\ref{sec:application}, với $21.000$ mẫu huấn luyện. Để phép đo công bằng, mọi mô hình chạy trên một luồng xử lý của cùng một máy tính, khi không có tiến trình nặng nào khác; tham số được chọn trước bằng kiểm định chéo trên một mẫu con $4.000$ điểm, và chỉ thời gian của một lần huấn luyện với tham số đã chọn được đo. Thời gian dự đoán là trung vị của bảy lần dự đoán toàn bộ tập kiểm tra, chia cho số mẫu. PWL-SVM dùng ánh xạ cộng tính với nút phân vị và $M/n=10$, theo quy trình rút ra ở Mục~\ref{sec:sensitivity}; thẻ điểm là hồi quy logistic trên các biến rời rạc hóa, như ở Mục~\ref{sec:application}.


\paragraph{Dự đoán và bộ nhớ.} Kết quả ở \cref{tab:cost-measured} và \cref{fig:cost} xác nhận dự báo lý thuyết. PWL-LS-SVM cộng tính dự đoán mỗi mẫu trong ${{T_LS_MAGIC}}$~$\mu$s trên Magic và ${{T_LS_CREDIT}}$~$\mu$s trên dữ liệu tín dụng, nhanh hơn SVM hạt nhân Gauss khoảng {{RT_MAGIC}} và {{RT_CREDIT}} lần. Chênh lệch đến từ cấu trúc của mô hình: trên dữ liệu tín dụng, SVM hạt nhân Gauss lưu ${{P_RBF_CREDIT}}$ số thực, ứng với khoảng ${{NSV_CREDIT}}$ vectơ hỗ trợ, và mỗi lần dự đoán phải tính chừng ấy khoảng cách cùng hàm mũ; còn PWL-SVM cộng tính chỉ lưu ${{P_LS_CREDIT}}$ số và thực hiện vài trăm phép so sánh, nhân và cộng, đúng như các công thức ở \cref{tab:cost}. {{COST_OTHERS}}

\input{chapters/tab_cost}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_cost.pdf}
\caption[Đánh đổi giữa độ chính xác và thời gian dự đoán]{Độ chính xác (Magic) hoặc AUC (dữ liệu tín dụng) theo thời gian dự đoán mỗi mẫu, trên thang logarit. Ô vuông là các biến thể PWL-SVM; điểm tròn là các phương pháp khác.}
\label{fig:cost}
\end{figure}

\paragraph{Huấn luyện.} {{TRAIN_LEAD}}: ${{TF_LS_MAGIC}}$ giây trên Magic và ${{TF_LS_CREDIT}}$ giây trên dữ liệu tín dụng, vì chỉ cần lập và giải một hệ tuyến tính cỡ $M\times M$ (\cref{prop:ridge}). PWL-C-SVM chậm hơn (${{TF_C_MAGIC}}$ và ${{TF_C_CREDIT}}$ giây) vì bộ giải lặp của LIBLINEAR cần nhiều vòng để hội tụ. SVM hạt nhân Gauss cần ${{TF_RBF_MAGIC}}$ và ${{TF_RBF_CREDIT}}$ giây, và chi phí này tăng nhanh hơn tuyến tính theo số mẫu~\cite{chang2011}; mạng nơ-ron cần ${{TF_MLP_MAGIC}}$ và ${{TF_MLP_CREDIT}}$ giây. Trong một quy trình thực tế, nơi mô hình phải được huấn luyện lại nhiều lần để chọn tham số, khác biệt này còn được nhân lên theo kích thước lưới tham số.

Cần lưu ý rằng số đo tuyệt đối phụ thuộc vào cài đặt: PWL-SVM được cài bằng NumPy, còn LIBSVM là thư viện C đã được tối ưu nhiều năm. Trên một thiết bị nhúng, số phép tính và số thực cần lưu ở \cref{tab:cost} mới là thước đo phù hợp, và ở cả hai thước đo đó, PWL-SVM cộng tính có lợi thế lớn so với SVM hạt nhân Gauss.
