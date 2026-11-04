# Phase-Gate Protocol

## Gate Sequence (per phase)
```
DRAFT (chat) → APPROVE → SAVE PLAN → EXECUTE → AUDIT → PRESENT → APPROVE → CHECKPOINT
```

## File Naming Convention
| Folder | Pattern | Example |
|---|---|---|
| `02_plans/` | `NN_PHASE_N_PLAN.md` | `01_PHASE_1_PLAN.md` |
| `03_checkpoints/` | `CHECKPOINT_NNN.md` | `CHECKPOINT_001.md` |
| `04_audits/` | `AUDIT_NNN.md` | `AUDIT_001.md` |
| `05_chat_logs/` | `LOG_NNN_topic.md` | `LOG_001_decisions.md` |

## Transition Triggers
- **Plan approved** → write to `02_plans/` → begin execution
- **Execution done** → run code → write audit → present to user → STOP
- **User says "Approved"** → write checkpoint → update `CURRENT_STATE.md` → draft next phase
- **User says "Initiate Rollback"** → revert git → update state → ask which checkpoint

