\subsection{Kết quả}\label{sec:credit-results}

\input{chapters/tab_credit}

\begin{figure}[t]
\centering
\includegraphics[width=0.62\linewidth]{fig_credit_roc.pdf}
\caption[Đường cong ROC trên dữ liệu tín dụng]{Đường cong ROC trên tập kiểm tra của lần chia đầu tiên cho bốn mô hình; con số trong ngoặc là AUC trung bình qua năm lần chia.}
\label{fig:credit-roc}
\end{figure}

\Cref{tab:credit} và \cref{fig:credit-roc} cho thấy các mô hình chia thành hai nhóm rõ rệt. Nhóm dẫn đầu gồm {{LEAD_TEXT}}. Nhóm còn lại gồm hồi quy logistic trên các biến gốc ({{A_LOGIT}}), SVM tuyến tính ({{A_LIN}}) và SVM hạt nhân Gauss ({{A_RBF}}); mạng nơ-ron ({{A_MLP}}) đứng giữa hai nhóm. Như vậy, AUC của PWL-LS-SVM cộng tính với nút phân vị {{SENT_BEST}}. Độ chính xác cân bằng và độ đo F1 ở ngưỡng đã chọn dẫn tới cùng kết luận: bốn mô hình của nhóm dẫn đầu gần như ngang nhau và cao hơn rõ rệt hồi quy logistic, SVM tuyến tính và SVM hạt nhân Gauss. Kết quả rất ổn định giữa các lần chia: độ lệch chuẩn của AUC chỉ khoảng {{SD_Q}}.

Khoảng cách {{D_Q_LOGIT}} đơn vị AUC giữa PWL-LS-SVM và hồi quy logistic trên các biến gốc cần được hiểu đúng. Phần lớn khoảng cách đó đến từ quan hệ phi tuyến giữa các trạng thái trả nợ và rủi ro (\cref{fig:credit-overview}a), mà một mô hình tuyến tính theo PAY\_0 không mô tả được. Trong thực tế, các thẻ điểm không dùng trực tiếp biến gốc mà rời rạc hóa từng biến thành các khoảng rồi mới hồi quy logistic~\cite{thomas2002}, và thẻ điểm xây dựng theo cách đó ở \cref{tab:credit} đã thu hẹp phần lớn khoảng cách. Thẻ điểm cũng là đối chứng công bằng nhất cho PWL-SVM cộng tính: cả hai đều là mô hình cộng tính, đều dựa trên các khoảng của từng biến, chỉ khác ở chỗ đóng góp của mỗi biến là hàm bậc thang trong thẻ điểm và là đường gấp khúc liên tục trong PWL-SVM. So với thẻ điểm, PWL-LS-SVM cao hơn ở {{N_Q_GT_BIN}}, với chênh lệch trung bình {{D_Q_BIN}} đơn vị AUC, trong khi hai mô hình có kích thước tương đương ({{P_Q}} và {{P_BIN}} số thực). Chênh lệch này nhỏ, nhưng trong chấm điểm tín dụng, ngay cả những cải thiện nhỏ về khả năng phân biệt cũng có thể mang lại lợi ích kinh tế đáng kể khi áp dụng cho một danh mục lớn~\cite{lessmann2015}.

Ba quan sát khác đáng chú ý. Thứ nhất, lưới phân vị tốt hơn lưới đều ({{A_Q}} so với {{A_UNI}}) với mô hình nhỏ hơn ({{P_Q}} so với {{P_UNI}} số thực), đúng như phân tích ở \cref{fig:credit-overview}b. Thứ hai, với cùng ánh xạ, PWL-C-SVM cho AUC thấp hơn hẳn PWL-LS-SVM ({{A_CQ}}). Điều này có thể giải thích bằng hàm mất mát: mất mát bản lề chỉ quan tâm đến các điểm nằm gần hoặc vi phạm lề, nên giá trị hàm quyết định ở xa biên không mang nhiều thông tin về mức độ rủi ro; còn mất mát bình phương của LS-SVM, tương đương hồi quy trên nhãn $\pm1$ (\cref{prop:ridge}), buộc điểm số của mọi khách hàng phải phản ánh nhãn của họ, và vì thế cho một thứ hạng tốt hơn. SVM hạt nhân Gauss, cũng dùng mất mát bản lề, gặp cùng hạn chế. Thứ ba, ánh xạ HH ({{A_HH}}) kém ánh xạ cộng tính: trên dữ liệu này, ảnh hưởng của từng biến quan trọng hơn tương tác giữa các biến, và các siêu phẳng ngẫu nhiên lãng phí khả năng biểu diễn, phù hợp với kết luận của Mục~\ref{sec:sensitivity}.

\subsection{Diễn giải mô hình}\label{sec:credit-interp}

Vì ánh xạ là cộng tính, hàm quyết định của PWL-LS-SVM tách thành tổng các đóng góp của từng biến,
\begin{equation}\label{eq:additive-decomp}
f(x)=w_0+\sum_{i=1}^{n}h_i\bigl(x(i)\bigr),
\end{equation}
trong đó mỗi $h_i$ là một hàm tuyến tính từng phần một biến, có không quá chín điểm gãy đặt tại các phân vị của biến đó (\cref{prop:1d}). Mỗi $h_i$ có thể được vẽ ra và đọc trực tiếp, giống như một thẻ điểm trong đó số điểm cộng thêm cho mỗi khách hàng là một đường gấp khúc theo giá trị của biến. \Cref{fig:credit-shapes} vẽ sáu hàm $h_i$ có ảnh hưởng lớn nhất của mô hình huấn luyện trên lần chia đầu tiên, sau khi trừ đi giá trị trung bình trên tập huấn luyện; giá trị dương nghĩa là rủi ro cao hơn một khách hàng trung bình.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_credit_shapes.pdf}
\caption[Các hàm đóng góp của mô hình PWL-SVM cộng tính]{Sáu hàm đóng góp $h_i$ có ảnh hưởng lớn nhất của PWL-LS-SVM cộng tính (nút phân vị) trên dữ liệu tín dụng, đã trừ giá trị trung bình. Nét đậm: mô hình của lần chia đầu tiên, nét chấm dọc là vị trí các nút của nó; nét mảnh: mô hình của bốn lần chia còn lại. Giá trị dương ứng với rủi ro vỡ nợ cao hơn.}
\label{fig:credit-shapes}
\end{figure}

{{SHAPES_TEXT}}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_credit_explain.pdf}
\caption[Mức độ ảnh hưởng của các biến và giải thích một hồ sơ]{(a) Mức độ ảnh hưởng của mười biến quan trọng nhất, đo bằng độ lệch chuẩn của đóng góp $h_i$ trên tập huấn luyện; với các biến định danh, đó là độ lệch chuẩn của tổng đóng góp của các biến chỉ báo tương ứng. (b) Tám đóng góp lớn nhất, so với khách hàng trung bình, của hồ sơ có điểm rủi ro cao nhất trong số các khách hàng vỡ nợ ở tập kiểm tra.}
\label{fig:credit-explain}
\end{figure}

{{EXPLAIN_TEXT}}

\subsection{Ràng buộc đơn điệu}\label{sec:monotone}

Người làm tín dụng kỳ vọng rằng rủi ro không giảm khi số tháng chậm trả tăng, và không tăng khi hạn mức tín dụng tăng. Mô hình không ràng buộc chỉ thỏa mãn một phần kỳ vọng này: ở {{VIOL_ANY}}, ít nhất một trong sáu hàm $h_i$ của các biến PAY có một đoạn với hệ số góc âm ở vùng từ trạng thái $0$ trở đi, và hàm của LIMIT\_BAL có đoạn đi lên ở {{VIOL_LIMIT}}. Như đã phân tích ở trên, đó là những đoạn được ước lượng từ rất ít dữ liệu, không phải là quy luật thật. Một mô hình trái với tri thức nghiệp vụ khó được chấp nhận khi thẩm định, dù độ chính xác của nó cao; vì vậy ta đưa kỳ vọng đơn điệu vào chính bài toán huấn luyện.

Với PWL-SVM cộng tính, điều này có thể làm mà vẫn giữ tính lồi. Gọi $t_{i1}<\dots<t_{iK}$ là các nút của biến thứ $i$ và $s_{i0},\dots,s_{iK}$ là hệ số góc của $h_i$ trên $K+1$ đoạn mà các nút chia ra. Khi đó
\begin{equation}\label{eq:slope-param}
h_i(t)=\sum_{j=0}^{K}s_{ij}\,G_{ij}(t),\qquad
G_{ij}(t)=\min\bigl\{(t-t_{ij})_+,\ t_{i,j+1}-t_{ij}\bigr\},
\end{equation}
với quy ước $t_{i0}=-\infty$, $t_{i,K+1}=+\infty$ (tức $G_{i0}(t)=\min\{t,t_{i1}\}$ và $G_{iK}(t)=(t-t_{iK})_+$): $G_{ij}(t)$ là độ dài phần đoạn thứ $j$ nằm bên trái $t$. Theo \cref{prop:1d}, hệ số của $x(i)$ trong ánh xạ~\eqref{eq:feature-map} bằng $s_{i0}$ và hệ số của $\bigl(x(i)-t_{ij}\bigr)_+$ bằng $s_{ij}-s_{i,j-1}$. Vì vậy~\eqref{eq:slope-param} chỉ là một cách tham số hóa khác của cùng một mô hình, và bài toán~\eqref{eq:ridge} trở thành
\[
\min_{s,\,w_0}\ \frac12\sum_{i=1}^{n}\Bigl[s_{i0}^2+\sum_{j=1}^{K}\bigl(s_{ij}-s_{i,j-1}\bigr)^2\Bigr]+\frac\gamma2\sum_{k=1}^{N}\Bigl(y_k-w_0-\sum_{i=1}^{n}h_i\bigl(x_k(i)\bigr)\Bigr)^2 .
\]
Hàm $h_i$ không giảm khi và chỉ khi $s_{ij}\ge0$ với mọi $j$, và không tăng khi và chỉ khi $s_{ij}\le0$ với mọi $j$; muốn $h_i$ đơn điệu chỉ trên một phần của trục, chẳng hạn trên nửa đường thẳng $t\ge t_{ij_0}$, ta chỉ cần áp các bất đẳng thức đó cho những đoạn nằm trong phần ấy. Thêm các ràng buộc này, ta được một bài toán bình phương tối thiểu với ràng buộc cận trên các biến --- một bài toán lồi, giải được chính xác bằng thuật toán BVLS~\cite{stark1995} có sẵn trong SciPy~\cite{virtanen2020}. Cũng như ở \cref{rem:convex}, tri thức tiên nghiệm được đưa vào mô hình mà không phải từ bỏ tính lồi. Khi không có ràng buộc nào, lời giải trùng với lời giải của \cref{prop:ridge} (sai khác dưới $10^{-9}$), điều được dùng để kiểm tra cài đặt.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_credit_monotone.pdf}
\caption[Hàm đóng góp khi có và không có ràng buộc đơn điệu]{Hàm đóng góp của PAY\_0, PAY\_4 và LIMIT\_BAL trong mô hình không ràng buộc (nét liền) và mô hình có ràng buộc đơn điệu (nét đứt), lần chia đầu tiên. Vùng tô xám: các trạng thái chậm trả từ ba tháng trở lên, chỉ chiếm {{SHARE_GE3}} số khách hàng.}
\label{fig:credit-monotone}
\end{figure}

Ta ràng buộc sáu hàm của các biến PAY không giảm từ trạng thái $0$ trở đi --- ba mã $-2$, $-1$ và $0$ chỉ những tình trạng không chậm trả khác nhau, không có thứ tự tự nhiên, nên được để tự do --- và hàm của LIMIT\_BAL không tăng trên toàn miền; các biến khác, các nút và hằng số $\gamma$ đã chọn cho mô hình không ràng buộc được giữ nguyên. \Cref{fig:credit-monotone} cho thấy tác động của ràng buộc: sau mức chậm hai tháng, đóng góp của PAY\_0 và PAY\_4 giữ nguyên thay vì giảm, và đường của LIMIT\_BAL trở thành đơn điệu giảm, trong khi hình dạng ở vùng có nhiều dữ liệu hầu như không đổi. {{MONO_EFFECT_LONG}}, và mô hình có ràng buộc vẫn {{MONO_VS_BIN}} thẻ điểm ({{A_BIN}}) (\cref{tab:credit}). Như vậy, với ràng buộc đơn điệu, PWL-LS-SVM cộng tính cho một mô hình vừa chính xác vừa nhất quán với tri thức nghiệp vụ.

\subsection{Khả năng triển khai}

Mô hình PWL-LS-SVM cộng tính trên dữ liệu tín dụng gồm {{P_Q}} số thực: các nút và hệ số của $20$ đường gấp khúc cho các biến số, hệ số của $6$ biến chỉ báo, hệ số chặn và hai vectơ chuẩn hóa. Để chấm điểm một hồ sơ, với mỗi biến ta chỉ cần xác định giá trị rơi vào đoạn nào giữa các nút rồi tính một hàm bậc nhất, tổng cộng vài trăm phép so sánh, nhân và cộng. Một mô hình như vậy có thể được cài đặt bằng vài dòng lệnh SQL trong kho dữ liệu của ngân hàng, bằng một bảng tính, hoặc trên thiết bị có tài nguyên rất hạn chế, mà không cần thư viện học máy nào; ràng buộc đơn điệu không làm thay đổi cấu trúc này. Việc huấn luyện lại khi có dữ liệu mới chỉ mất chưa tới một giây (\cref{tab:cost-measured}). Để so sánh, SVM hạt nhân Gauss cần lưu khoảng {{P_RBF}} số thực, và tổ hợp cây tăng cường gồm từ {{HGB_TREES_MIN}} đến {{HGB_TREES_MAX}} cây quyết định với khoảng {{P_HGB}} tham số; cả hai đều khó kiểm tra và giải thích hơn nhiều.
