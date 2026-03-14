"""
开发板推理主脚本 (classify_test.py)
基于MobileNetV2的垃圾分类 - 在 Atlas 200DK 上使用 AscendCL 执行推理
运行方式：python3.6 src/classify_test.py ./data/
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import acl

# ---- AscendCL 常量 ----
ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_MEMCPY_HOST_TO_DEVICE  = 1
ACL_MEMCPY_DEVICE_TO_HOST  = 2
MEMORY_DEVICE = 1
SUCCESS = 0

# ---- 垃圾分类标签（与训练时保持一致）----
image_net_classes = [
    'Plastic Bottle', 'Hats', 'Newspaper', 'Cans', 'Glassware',
    'Glass Bottle', 'Cardboard', 'Basketball', 'Paper', 'Metalware',
    'Disposable Chopsticks', 'Lighter', 'Broom', 'Old Mirror', 'Toothbrush',
    'Dirty Cloth', 'Seashell', 'Ceramic Bowl', 'Paint bucket', 'Battery',
    'Fluorescent lamp', 'Tablet capsules', 'Orange Peel', 'Vegetable Leaf',
    'Eggshell', 'Banana Peel',
]


def get_image_net_class(label_id):
    """根据类别索引返回英文类别名称。"""
    if label_id >= len(image_net_classes):
        return "unknown"
    return image_net_classes[label_id]


def check_ret(message, ret):
    """断言 AscendCL 调用返回值正常。"""
    if ret != SUCCESS:
        raise Exception("{} failed, ret={}".format(message, ret))


# ============================================================
# AclModel：封装模型加载、推理、资源释放
# ============================================================
class AclModel:
    def __init__(self, model_path):
        self.model_path   = model_path
        self.model_id     = None
        self.model_desc   = None
        self.input_dataset  = None
        self.output_dataset = None
        self.init_resource()

    def init_resource(self):
        """加载 .om 模型文件，构建输出内存。"""
        print("[Model] class Model init resource stage:")
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)

        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self._gen_output_dataset(output_size)
        self._get_output_info(output_size)
        print("[Model] class Model init resource stage success")
        return SUCCESS

    def _get_output_info(self, output_size):
        self.output_info = []
        for i in range(output_size):
            dims, ret = acl.mdl.get_output_dims(self.model_desc, i)
            datatype   = acl.mdl.get_output_data_type(self.model_desc, i)
            self.output_info.append({"dims": dims["dims"], "datatype": datatype})

    def _gen_output_dataset(self, size):
        """为每个输出节点分配设备内存，构建 output_dataset。"""
        print("[Model] create model output dataset:")
        dataset = acl.mdl.create_dataset()
        for i in range(size):
            temp_buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY)
            check_ret("acl.rt.malloc", ret)
            dataset_buffer = acl.create_data_buffer(temp_buffer, temp_buffer_size)
            acl.mdl.add_dataset_buffer(dataset, dataset_buffer)
        self.output_dataset = dataset
        print("[Model] create model output dataset success")

    def _gen_input_dataset(self, data, data_size):
        """构建模型输入数据集。"""
        self.input_dataset = acl.mdl.create_dataset()
        input_dataset_buffer = acl.create_data_buffer(data, data_size)
        acl.mdl.add_dataset_buffer(self.input_dataset, input_dataset_buffer)

    def execute(self, data, data_size):
        """执行模型推理，返回 numpy 格式的输出列表。"""
        self._gen_input_dataset(data, data_size)
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        check_ret("acl.mdl.execute", ret)

        inference_result = []
        num_outputs = acl.mdl.get_dataset_num_buffers(self.output_dataset)
        for i in range(num_outputs):
            buf      = acl.mdl.get_dataset_buffer(self.output_dataset, i)
            data_ptr = acl.get_data_buffer_addr(buf)
            size     = acl.get_data_buffer_size(buf)
            output_host, _ = acl.rt.memcpy(
                None, size, data_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST
            )
            dims  = self.output_info[i]["dims"]
            dtype = np.float32
            arr   = np.frombuffer(output_host, dtype=dtype).reshape(dims)
            inference_result.append(arr)
        return inference_result

    def __del__(self):
        self._release_dataset()
        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)
            check_ret("acl.mdl.destroy_desc", ret)
        print("Model release source success")

    def _release_dataset(self):
        for dataset in [self.input_dataset, self.output_dataset]:
            if not dataset:
                continue
            num = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(num):
                buf = acl.mdl.get_dataset_buffer(dataset, i)
                if buf:
                    acl.destroy_data_buffer(buf)
            acl.mdl.destroy_dataset(dataset)


# ============================================================
# Sample：整合 ACL 初始化 / DVPP 预处理 / 推理 / 后处理
# ============================================================
class Sample:
    def __init__(self, model_path):
        self._model    = None
        self._dvpp     = None
        self._run_mode = None
        self._init_resource(model_path)

    def _init_resource(self, model_path):
        """初始化 ACL 系统资源、设备、上下文。"""
        print("[Sample] init resource stage:")
        ret = acl.init()
        check_ret("acl.init", ret)
        ret = acl.rt.set_device(0)
        check_ret("acl.rt.set_device", ret)
        self._context, ret = acl.rt.create_context(0)
        check_ret("acl.rt.create_context", ret)
        self._run_mode, ret = acl.rt.get_run_mode()
        check_ret("acl.rt.get_run_mode", ret)

        from atlas_utils.acl_dvpp import Dvpp
        self._dvpp  = Dvpp()
        self._model = AclModel(model_path)
        print("[Sample] init resource stage success")

    def _pre_process(self, image_file):
        """使用 DVPP 对图像进行 JPEG 解码 + Resize，返回 YUV420SP 格式图像。"""
        from atlas_utils.acl_image import AclImage
        image       = AclImage(image_file)
        image_dvpp  = self._dvpp.jpegd(image)           # JPEGD 解码 -> YUV420SP
        resized     = self._dvpp.resize(image_dvpp, 224, 224)  # Resize 到 224x224
        return resized

    def post_process(self, infer_output, image_file):
        """解析推理输出，取 Top-5，将最高置信度类别标注在图片上保存。"""
        print("post process")
        data   = infer_output[0]
        vals   = data.flatten()
        top_k  = vals.argsort()[-1:-6:-1]   # 取 Top-5 索引

        print("images: {}".format(image_file))
        print("======== top5 inference results: =============")
        for n in top_k:
            obj_cls = get_image_net_class(n)
            print("label: %d  confidence: %f  class: %s" % (n, vals[n], obj_cls))

        if len(top_k):
            obj_cls     = get_image_net_class(top_k[0])
            output_path = os.path.join("./outputs", os.path.basename(image_file))
            origin_img  = Image.open(image_file)
            draw = ImageDraw.Draw(origin_img)
            draw.text((10, 10), obj_cls, fill=(0, 255, 0))
            origin_img.save(output_path)

    def run(self, image_dir):
        """遍历图像目录，依次执行推理并保存结果。"""
        os.makedirs("./outputs", exist_ok=True)
        images = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        for image_file in images:
            image     = self._pre_process(image_file)
            result    = self._model.execute(image.data(), image.size)
            self.post_process(result, image_file)

    def __del__(self):
        if self._model:
            del self._model
        if self._dvpp:
            del self._dvpp
        acl.rt.destroy_context(self._context)
        acl.rt.reset_device(0)
        acl.finalize()
        print("[Sample] class Sample release source success")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3.6 src/classify_test.py <image_dir>")
        sys.exit(1)

    MODEL_PATH = "../model/garbage_yuv.om"
    image_dir  = sys.argv[1]

    sample = Sample(MODEL_PATH)
    sample.run(image_dir)
