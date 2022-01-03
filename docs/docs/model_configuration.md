# Model Configuration
Mặc định, cấu hình phải định nghĩa trước cho mô hình các thông số như tên model, platform sử dụng (```tensorrt_plan, pytorch_libtorch, tensorflow_savedmodel, ...```), kiểu dữ liệu, kích thước cho input, output, cấu hình wramup, cấu hình optimization, ...
### 1. Cấu hình cơ bản (minimal model configuration)
Mặc định ta không cần xây dựng cấu hình cho các model TensorRT, Tensorflow saved-model và ONNX vì Triton có thể tự động generate. Đối với các model này nếu như không tồn tại ```config.pbtxt``` và ta khởi động triton-server với tham số ```--strict-model-config = false```, triton-server sẽ tự động generate ra file ```config.pbtxt``` ở mức cơ bản. Còn lại ta sẽ phải xây dựng file ```config.pbtxt``` bằng tay. Ở đây mình sẽ xây dựng cấu hình cho model GFPGan đã được training sử dụng pytorch.
```

```

<i>Giá trị **-1** thể hiện cho **dynamic-shape** </i>

Câu hỏi đặt ra là làm thế nào ta biết được các inputs cũng như outputs kèm các thông số của nó như kiểu dữ liệu, kích thước ? Cái này thì mình nghĩ mọi người làm việc đủ sâu với model tự khắc sẽ biết cách thôi.