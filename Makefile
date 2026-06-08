PYTHON ?= python3
POOL ?= data/gen9_random_pool.json
REPLAYS ?= data/replays
ANNOTATIONS ?= data/annotations
CHECKPOINTS ?= checkpoints

.PHONY: compile pool phase0 validate-phase0 phase1-smoke smoke ssh-probe

compile:
	$(PYTHON) -m compileall belief mcts pokenet phase0 rlm training agent eval

pool:
	$(PYTHON) -m data.fetch_gen9_random_pool --output $(POOL)

phase0:
	$(PYTHON) -m phase0.rlm_annotator --replays $(REPLAYS) --output $(ANNOTATIONS) --pool $(POOL)

validate-phase0:
	$(PYTHON) -m phase0.validate_annotations --annotations $(ANNOTATIONS) --pool $(POOL)

phase1-smoke:
	$(PYTHON) -m training.phase1_il --annotations $(ANNOTATIONS) --pool $(POOL) --output $(CHECKPOINTS)/phase1_smoke.pt --epochs 1 --batch-size 8 --max-decisions 32

smoke: compile phase0 validate-phase0 phase1-smoke
	$(PYTHON) -m agent.showdown_agent --mode smoke

ssh-probe:
	ssh -i ~/bleh.pem -o BatchMode=yes -o ConnectTimeout=10 ubuntu@98.91.244.123 'uname -a && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
