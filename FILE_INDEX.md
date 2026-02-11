# Collective Memory 文件索引

快速导航项目文件结构。

---

## 核心文档

| 文件 | 说明 |
|------|------|
| `CLAUDE.md` | 项目总文档（理论、实现、会议记录、经验教训） |
| `FILE_INDEX.md` | 本文件 |

---

## REM 模拟代码

### 核心代码 (`Simulations/src/`)

| 文件 | 说明 | 关键函数 |
|------|------|----------|
| `rem_core.py` | REM模型核心引擎（**只读**） | `compute_log_odds()` |
| `group_rules.py` | 7种群体决策规则 | `CF`, `UW`, `DMC`, `DSS`, `BF`, `UW_Miscal`, `DMC_Miscal`, `rich_conflict_rule` |
| `run_simulation.py` | 参数扫描主程序 | `run_bahrami_sweep()`, `run_miscalibration_sweep()`, `run_rich_conflict_simulation()` |

### Jupyter Notebooks (`Simulations/notebooks/`)

| 文件 | 说明 |
|------|------|
| `bahrami_sweep_demo.ipynb` | Bahrami参数扫描交互式演示 |

### 输出文件 (`Simulations/outputs/`)

| 文件 | 说明 |
|------|------|
| `bahrami_sweep_final.csv` | Bahrami sweep 结果（45行：9 c_B × 5 rules） |
| `miscalibration_sweep.csv` | Miscalibration sweep 结果（44行：11 α_B × 4 models） |
| `rich_conflict_results.csv` | Rich冲突模型验证结果 |
| `*.png` | 可视化图表 |

### 学习资源 (`Simulations/learning_materials/`)

| 文件 | 说明 | 状态 |
|------|------|------|
| `REM-2FC.py` | 复杂2FC REM模拟（5000列表，3种刺激） | 参考用 |
| `REM-generic.py` | 简洁REM学习版本（100项目） | 参考用 |
| `R codes to yiyan.r` | R数据分析和ggplot可视化 | 参考用 |

**注意**：学习资源非正式项目代码，仅供参考。

---

## 在线实验系统

**位置**：`/Users/yiytan/memory-game/`（独立目录）

| 路径 | 说明 |
|------|------|
| `src/App.js` | 核心实验逻辑（1481行） |
| `src/index.js` | React入口 |
| `build/` | 生产构建输出 |
| `package.json` | 项目配置 |
| `firebase.json` | Firebase Hosting配置 |
| `.firebaserc` | Firebase项目关联 |

**部署地址**：https://collective-memory-d3802.web.app

---

## IRB 文档

**位置**：`IRB/`

| 文件 | 说明 |
|------|------|
| `Protocols*.pdf` | IRB反馈（只读） |
| `*_REVISED.docx` | 修订后的IRB文档（6个） |
| `IRB_REVISION_INSTRUCTIONS.txt` | 修订使用说明 |

---

## 其他目录

| 目录 | 说明 |
|------|------|
| `papers/` | 相关论文PDF |
| `data/` | 模拟数据归档 |
| `experiments/` | 实验设计文档 |
| `docs/` | 会议记录、邮件存档 |
| `Simulations/archive/` | 遗留代码归档 |

---

## 快速定位

### 按任务查找

| 我想... | 去哪里 |
|---------|--------|
| 理解REM公式 | `CLAUDE.md` → 共享理论基础 |
| 运行参数扫描 | `Simulations/src/run_simulation.py` |
| 修改决策规则 | `Simulations/src/group_rules.py` |
| 测试在线实验 | `/Users/yiytan/memory-game/` → `npm start` |
| 部署实验 | `/Users/yiytan/memory-game/` → `npm run build && firebase deploy` |
| 查看模拟结果 | `Simulations/outputs/` |
| 修改IRB文档 | `IRB/` |
| 学习REM基础 | `Simulations/learning_materials/REM-generic.py` |

### 按导师需求查找

| 导师 | 可能需要 | 文件位置 |
|------|----------|----------|
| Rich | 冲突模型验证 | `group_rules.py:rich_conflict_rule()` |
| Tim | Bahrami/Miscalibration结果 | `outputs/bahrami_sweep_final.csv`, `outputs/miscalibration_sweep.csv` |
| Steve | 实验设计细节 | `CLAUDE.md` → 分支4: Online Experiment |

---

*最后更新：2026年2月11日*
