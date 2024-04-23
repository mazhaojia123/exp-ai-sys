from setuptools import setup
from torch.utils import cpp_extension

setup(
    #TODO: 给出编译后的链接库名称
    name='hsigmoid_module',
    ext_modules=[
        cpp_extension.CppExtension(
            'hsigmoid_module',
            ['hsigmoid.cpp']
        )
    ],
    # 执行编译命令设置
    cmdclass={						       
        'build_ext': cpp_extension.BuildExtension
    }
)
print("generate .so PASS!\n")