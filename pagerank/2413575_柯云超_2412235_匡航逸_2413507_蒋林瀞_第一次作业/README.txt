PageRank 第一次作业提交包

小组成员：
- 2413575 柯云超（B）
- 2412235 匡航逸（A）
- 2413507 蒋林瀞（C）

目录说明：
- 源码：Python 源码、冻结接口、实验脚本、测试用例和打包参数。
- 实验结果：beta=0.85、eps=1e-8、csr_block+K=8+float32 的 Res.txt，以及实验摘要。
- 实验报告：正式报告 PDF。
- 可执行文件：Windows 单文件可执行程序 main.exe。

运行方式：

1. 运行 Windows 可执行文件（无需 Python 环境）：
   可执行文件/main.exe --data Data.txt --out Res.txt --mode csr_block --K 8 --dtype float32 --beta 0.85 --eps 1e-8

2. 运行源码（需 Python 3 + numpy + scipy）：
   pip install numpy scipy psutil
   python 源码/main.py --data Data.txt --out Res.txt --mode csr_block --K 8 --dtype float32 --beta 0.85 --eps 1e-8

3. Linux / macOS 打包为可执行文件：
   pip install pyinstaller
   cd 源码
   pyinstaller --noconfirm --clean --onefile --name main main.py --exclude-module matplotlib --exclude-module tkinter --exclude-module PIL --exclude-module pytest
   生成的可执行文件位于 源码/dist/main。

注意：
- Data.txt 已包含在提交包根目录，可直接运行。
- main.exe 仅适用于 Windows；Linux / macOS 请使用源码运行或自行打包。
- 源码/scripts/memoryuse-python.py 是老师提供的内存测量辅助脚本，保留用于人工复核。
