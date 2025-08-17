# LLM-Quant-Factory


---

## 🚀 功能特性
1.本项目主要基于Sigma-Delta ADC思想，设计了一种面向大模型的Sigma-Delta量化方案，该量化方法可将大模型参数量化至1/1.5bit。
2.本项目支持多种支持多种主流量化方案的伪量化实现，便于科研复现。


---

## 📦 安装与环境配置
确保你已经安装了 Python 3.x。

```bash
git clone https://github.com/Dreamlittlecat/LLM-Quant-Factory.git
cd LLM-Quant-Factory
pip install -r requirements.txt
```
示例用法：

```bash

cd LLM-Quant-Factory
#模型量化
bash run_example.sh
#模型zero_shot评测
bash run_zeroshot.sh
```