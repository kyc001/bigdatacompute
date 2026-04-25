PageRank 第一次作业运行说明

一、文件说明

main.py
  Python 主入口，支持 dense / csr / block / csr_block 四种模式。

blocks.py
  分块矩阵相关实现，main.py 的 csr_block 模式依赖该文件。

compile-parameter.txt
  PyInstaller 打包命令。

main.exe
  Windows 单文件可执行程序，由 PyInstaller 生成。

Res.txt
  beta = 0.85、eps = 1e-8、csr_block + K=8 + float32 下的 Top-100 PageRank 结果。

Report.pdf
  实验报告。

二、源码运行命令

python main.py --data Data.txt --beta 0.85 --eps 1e-8 --out Res.txt --mode csr_block --K 8 --dtype float32

三、可执行文件运行命令

main.exe --data Data.txt --beta 0.85 --eps 1e-8 --out Res.txt --mode csr_block --K 8 --dtype float32

四、输出格式

Res.txt 每行格式为：

NodeID Score

其中 Score 保留 10 位小数，输出按 Score 降序排列；Score 相同时按 NodeID 升序排列。

五、实验环境

开发与验证环境为 Windows + Python 3.12。主要依赖包括 numpy、scipy、psutil、pytest、pyinstaller、pandas、matplotlib。
