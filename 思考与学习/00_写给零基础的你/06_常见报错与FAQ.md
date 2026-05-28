# 常见报错与 FAQ

> 5 分钟扫一遍, 知道大概会卡哪里。
> 遇到报错时回来 Ctrl+F 搜关键词就行。

本篇专门收"小白阶段最常踩的坑", 进阶的数据相关问题（yfinance 限流、akshare 代理等）见 [05 章 / 数据源问题排查](../05_扩展指南/数据源问题排查.md)。

---

## 一、安装相关

### `python: command not found` / `'python' 不是内部或外部命令`

原因是 Python 没装, 或者装了但 PATH 没配。

怎么办:
- Windows: 重装 Python, 务必勾选 "Add Python to PATH"
- Mac: `brew install python@3.11`, 然后用 `python3` 而不是 `python`
- 验证: `python --version` (Mac 试 `python3 --version`)

如果你已经装了 Python 但还不行, 重启电脑试试（Windows 上 PATH 改了要重登）。

---

### `pip: command not found`

原因是 pip 没装好或 PATH 没配。

```bash
python -m ensurepip --upgrade
```
或者直接用 `python -m pip install ...` 代替 `pip install ...`。

---

### `pip install` 卡住 / 极慢

默认源在国外, 国内网络很慢。换清华镜像:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

永久配置:
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### PowerShell 报"无法加载脚本, 因为禁用脚本运行"

Windows 默认禁止跑 .ps1 脚本（包括激活虚拟环境的脚本）。用管理员身份打开 PowerShell, 跑:
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```
选 Y, 然后重开 PowerShell, 再试激活虚拟环境。

---

### 虚拟环境激活后, `pip install` 还是装到全局

多半是你打开了一个新终端但没激活 venv。看命令行最前面有没有 `(.venv)` 标识, 没有就重新激活:
```bash
# Windows
.venv\Scripts\Activate.ps1

# Mac/Linux
source .venv/bin/activate
```

---

## 二、运行相关

### `ModuleNotFoundError: No module named 'pandas'` (或其他库)

99% 是虚拟环境没激活, 库装到了别的环境去了。

1. 看命令行最前面有没有 `(.venv)`, 没有就激活
2. 激活后再 `pip install -r requirements.txt` 装一遍
3. 重跑你的脚本

如果激活了还报错, 说明虚拟环境装的 Python 跟你以为的不是同一个, 检查:
```bash
which python      # Mac/Linux
where python      # Windows
```
应该指向 `.venv/Scripts/python` 或 `.venv/bin/python`。

---

### `FileNotFoundError: examples/00_xxx.py`

你不在项目根目录, 而是在别的地方跑命令。

```bash
cd /你的桌面路径/quant-framework    # 先进根目录
python examples/00_模拟数据_全策略对比.py
```

可以 `pwd` (Mac) 或 `cd` (Windows) 看当前目录是不是 `quant-framework`。

---

### 中文乱码 (一堆问号 ?????? 或 UnicodeEncodeError)

Windows 命令行默认编码是 GBK, 我们脚本里有中文输出, 直接打就乱码。设环境变量再跑:
```bash
# Windows PowerShell
$env:PYTHONIOENCODING="utf-8"
python examples/00_模拟数据_全策略对比.py

# Windows cmd
set PYTHONIOENCODING=utf-8
python examples/00_模拟数据_全策略对比.py

# Mac/Linux (通常不用设)
PYTHONIOENCODING=utf-8 python examples/00_模拟数据_全策略对比.py
```

想一劳永逸的话（推荐）: 改 Windows 控制面板 → 区域 → 管理 → 更改系统区域设置 → 勾选 "Beta: 使用 Unicode UTF-8" → 重启。
此后命令行原生支持中文。

---

### `ImportError: cannot import name 'xxx' from 'pandas'`

pandas 版本不匹配。本框架要求 pandas >= 2.0, 你装的可能太老。
```bash
pip install -U pandas numpy scikit-learn
```

---

### 跑 examples/00 一直转圈不出结果

第一次跑会编译 numpy 的 C 后端, 卡 5~10 秒正常。
如果超过 30 秒还没动静, Ctrl+C 中止, 检查代码有没有改坏（特别是改了循环条件）。

---

## 三、Git 相关

### `git clone` 提示输密码 / 失败

仓库是私有的, 你没访问权限。三个办法:
1. 让作者把你加为 collaborator（GitHub 仓库 Settings → Collaborators）
2. 或者 fork 到你自己账号下再 clone 你 fork 的版本
3. 或者直接在 GitHub 网页 "Code → Download ZIP"

---

### GitHub 克隆很慢或失败

国内访问 github 容易抽风, 可以这样:
1. 走代理（如果你有 Clash/V2Ray）:
   ```bash
   git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 clone <url>
   ```
   端口换成你代理软件的实际端口。

2. 用镜像 (社区镜像不稳定, 仅作备用):
   ```bash
   git clone https://hub.fastgit.xyz/XinghaoCui/quant-framework.git
   # 或
   git clone https://gitclone.com/github.com/XinghaoCui/quant-framework.git
   ```

3. 从 Gitee 镜像 (如果作者上传了的话)

4. 离线方案: 让作者打包 zip 发你

---

### push 时报 "Could not resolve host github.com"

git 不会自动走代理, 但你的网络需要代理才能访问 github。跟克隆一样, 用 `-c` 参数走代理:
```bash
git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 push
```

或者全局配置:
```bash
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
```
这样配了之后, 推国内 git 服务时记得 unset, 否则会反过来报错。

---

## 四、数据相关

### yfinance 报 "Too Many Requests"
见 [05 章 / 数据源问题排查](../05_扩展指南/数据源问题排查.md#错误-1yfinance-too-many-requests).

### akshare 报代理错误
见 [05 章 / 数据源问题排查](../05_扩展指南/数据源问题排查.md#错误-2akshare-代理错误).

### 数据时间不对 / 数据缺失
见 [05 章 / 数据源问题排查](../05_扩展指南/数据源问题排查.md#错误-3数据缺失或时间不对).

小白阶段的建议是: 不要纠结真实数据问题, 全程跑 `examples/00`（模拟数据）就行, 等以后再处理。

---

## 五、概念性 FAQ

### Q: 我跑出来的夏普 1.69, 是真的赚钱了吗?
那是模拟数据上的回测结果, 不是真钱。模拟数据是程序随机生成的, 不存在真实赚钱可能。
真要赚钱, 得跑真实数据, 而且真实策略夏普通常远低于模拟数据。

### Q: 为什么动量策略夏普这么高 (1.73), 我能去 A 股复制吗?
不能, 几个原因:
- 模拟数据没有反映真实市场的"动量衰减"
- A 股有涨跌停、T+1 限制, 简单动量策略实际执行起来很难
- 真实交易成本（佣金 + 印花税 + 滑点）会吃掉一大半收益

把这个仓库当学方法的工具, 不要当赚钱秘籍。

### Q: 跑出来的图在哪?
项目根目录的 `reports/` 文件夹下, 每个策略一个子目录。

### Q: 我能跑实盘吗?
不能, 本框架是纯回测框架, 不带下单接口。实盘要接券商 API（如 vn.py、QMT、QuickTrade）, 那是另一套系统。

### Q: examples 编号有什么规律?
- `00`: 模拟数据全策略对比（入门）
- `01-09`: 真实数据基础策略
- `10-19`: 机器学习相关
- `20-29`: 因子研究相关
- 以后还会加 `30-39` 组合优化、`40-49` 风险归因等

### Q: 我要先学 pandas / Python 到什么程度?
见 [07 完全没编程基础怎么办](./07_完全没编程基础怎么办.md), 用不着精通, 能看懂示例就行。

### Q: 我看不懂代码里的 `from __future__ import annotations`, 是不是没学过这玩意?
不影响理解, 可以无视。它是 Python 类型提示的兼容性写法, 不改变功能。

### Q: 文档里出现的 `MultiIndex` / `corrwith` / `groupby` 是啥?
都是 pandas 的概念。第一次遇到不需要立刻搞懂, 跟着文档说的"做什么"就行, 用多了自然懂。急着学就去 pandas 官方文档 quickstart。

---

## 六、还有问题怎么办

按这个顺序找:
1. 本篇 FAQ, 你正在看
2. [05 章 / 数据源问题排查](../05_扩展指南/数据源问题排查.md), 数据相关都在这
3. 报错信息的最后一行: Python 报错一般最后一行最关键, 把它复制到 Google / 必应搜
4. 直接复制报错给 ChatGPT/Claude, AI 在解释报错上意外好用
5. 看代码的 docstring, 本框架每个模块都有中文 docstring 讲清"为什么这么写"
6. 去 GitHub issue 提问, 仓库主页可以提 issue, 作者会回

---

## 一句话总结

> 报错不是你菜, 是所有人共同经历的过程。拍下来发给 AI, 90% 能解决。

走吧, 最后一篇 → [07 完全没编程基础怎么办](./07_完全没编程基础怎么办.md)
