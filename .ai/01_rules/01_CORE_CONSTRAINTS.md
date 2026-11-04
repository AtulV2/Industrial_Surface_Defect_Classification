# Core Constraints

Loaded from: `ai.md` (2026-11-04)

1. **Immutable Logging** — Never overwrite past plans/audits/logs. Always create new sequentially numbered files.
2. **Record Everything** — Log all significant decisions in `05_chat_logs/` before ending a session.
3. **Draft Before Writing** — Present plan in chat, await approval before writing to file system.
4. **Iterate and Lock** — Save approved plan as immutable file in `02_plans/`.
5. **Execute Phase** — Only write code after plan is saved.
6. **Audit and Verify** — Run code, then create audit log in `04_audits/`.
7. **Strict Stop and Wait** — Do NOT auto-proceed to next phase. Present audit and ask permission.
8. **Lock and Checkpoint** — Only after "Approved": write checkpoint, update `CURRENT_STATE.md`, draft next phase.
9. **Continuous Git Versioning** — Provide `git add` and `git commit` commands after every working sub-step.
10. **Human-Readable Code** — Clean, modular, commented code explaining the *why*.
11. **Repository Onboarding Files** — Maintain `README.md`, `CONTRIBUTING.md` in project root.
12. **Session Pause and Handoff** — Save pending draft to `PENDING_NEXT_PHASE_DRAFT.md` on pause.
13. **Optimized Context Loading** — Read in strict order: Entry → Rules → State → Checkpoint → Pending.
14. **No Blind Edits** — Always read a file's current contents before modifying it.
15. **Dependency Tracking** — Update `requirements.txt` and audit log on every new dependency.
16. **Error Loop Escalation** — Stop after 3 failed fix attempts and escalate.
17. **Anti-Scope Creep** — Strictly confine changes to current phase only.
18. **Zero Secrets Policy** — No hardcoded keys/passwords. Use `.env` + `.gitignore`.
19. **Execution Validation** — Run code/tests before presenting audit report.
20. **Rollback Protocol** — On "Initiate Rollback": stop, revert to last commit, update state.

