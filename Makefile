IMAGE_NAME = equihgnn

# OPV task:
# molecular: 0-gap, 1-homo, 2-lumo, 3-spectral_overlap
# polymer: 4-homo, 5-lumo, 6-gap, 7-optical_lumo

#QM9 task:
# 0-alpha, 1-gap, 2-homo, 3-lumo, 4-mu, 5-cv
TASK_ID ?= $(word 2, $(MAKECMDGOALS))

OPV_TASK_ID_RANGE = $(shell seq 0 7)
QM9_TASK_ID_RANGE = $(shell seq 0 5)

build:
	@echo "Checking if Docker image $(IMAGE_NAME) exists..."
	@if docker images -q $(IMAGE_NAME) > /dev/null 2>&1; then \
	    echo "Checking if the Docker image $(IMAGE_NAME) is being used by any containers..."; \
	    if docker ps -q --filter "ancestor=$(IMAGE_NAME)" | grep .; then \
	        echo "The image $(IMAGE_NAME) is in use by running containers. Please stop them before removing the image."; \
	        exit 1; \
	    fi; \
	    echo "Removing existing Docker image $(IMAGE_NAME)..."; \
	    docker rmi $(IMAGE_NAME) || echo "Failed to remove image $(IMAGE_NAME). It may be in use."; \
	fi; \
	@echo "Building the Docker image..."; \
	docker build -t $(IMAGE_NAME) .


clean:
	@echo "Stopping and removing containers using Docker image $(IMAGE_NAME)..."
	@docker ps -q --filter "ancestor=$(IMAGE_NAME)" | xargs --no-run-if-empty docker stop
	@docker ps -a -q --filter "ancestor=$(IMAGE_NAME)" | xargs --no-run-if-empty docker rm


########## Train OPV ##########
train_opv:
	@docker run \
		--gpus all \
		-v $(shell pwd)/datasets:/equihgnn/datasets \
		-v $(shell pwd)/logs:/equihgnn/logs \
		-v $(shell pwd)/scripts:/equihgnn/scripts \
		-e COMET_API_KEY=$(COMET_API_KEY) \
		$(IMAGE_NAME) bash scripts/run_opv.sh $(TASK_ID)


train_opv_all:
	@echo "Starting training for TASK_ID range 0 to 7..."
	@for id in $(OPV_TASK_ID_RANGE); do \
		echo "Running task with TASK_ID=$$id"; \
		docker run \
			--gpus all \
			-v $(shell pwd)/datasets:/equihgnn/datasets \
			-v $(shell pwd)/logs:/equihgnn/logs \
			-v $(shell pwd)/scripts:/equihgnn/scripts \
			-e COMET_API_KEY=$(COMET_API_KEY) \
			$(IMAGE_NAME) bash scripts/run_opv.sh $$id; \
	done
	@echo "All training tasks completed."


########## Train OPV-3D ##########
train_opv3d:
	@docker run \
		--gpus all \
		-v $(shell pwd)/datasets:/equihgnn/datasets \
		-v $(shell pwd)/logs:/equihgnn/logs \
		-v $(shell pwd)/scripts:/equihgnn/scripts \
		-e COMET_API_KEY=$(COMET_API_KEY) \
		$(IMAGE_NAME) bash scripts/run_opv_3d.sh $(TASK_ID)


train_opv3d_all:
	@echo "Starting training for TASK_ID range 0 to 7..."
	@for id in $(OPV_TASK_ID_RANGE); do \
		echo "Running task with TASK_ID=$$id"; \
		docker run \
			--gpus all \
			-v $(shell pwd)/datasets:/equihgnn/datasets \
			-v $(shell pwd)/logs:/equihgnn/logs \
			-v $(shell pwd)/scripts:/equihgnn/scripts \
			-e COMET_API_KEY=$(COMET_API_KEY) \
			$(IMAGE_NAME) bash scripts/run_opv_3d.sh $$id; \
	done
	@echo "All training tasks completed."


########## Train QM9 ##########
train_qm9:
	@docker run \
		--gpus all \
		-v $(shell pwd)/datasets:/equihgnn/datasets \
		-v $(shell pwd)/logs:/equihgnn/logs \
		-v $(shell pwd)/scripts:/equihgnn/scripts \
		-e COMET_API_KEY=$(COMET_API_KEY) \
		$(IMAGE_NAME) bash scripts/run_qm9.sh $(TASK_ID)


train_qm9_all:
	@echo "Starting training for TASK_ID range 0 to 5..."
	@for id in $(QM9_TASK_ID_RANGE); do \
		echo "Running task with TASK_ID=$$id"; \
		docker run \
			--gpus all \
			-v $(shell pwd)/datasets:/equihgnn/datasets \
			-v $(shell pwd)/logs:/equihgnn/logs \
			-v $(shell pwd)/scripts:/equihgnn/scripts \
			-e COMET_API_KEY=$(COMET_API_KEY) \
			$(IMAGE_NAME) bash scripts/run_qm9.sh $$id; \
	done
	@echo "All training tasks completed."


########## Train QM9-3D ##########
train_qm9_3d:
	@docker run \
		--gpus all \
		-v $(shell pwd)/datasets:/equihgnn/datasets \
		-v $(shell pwd)/logs:/equihgnn/logs \
		-v $(shell pwd)/scripts:/equihgnn/scripts \
		-e COMET_API_KEY=$(COMET_API_KEY) \
		$(IMAGE_NAME) bash scripts/run_qm9_3d.sh $(TASK_ID)


train_qm9_3d_all:
	@echo "Starting training for TASK_ID range 0 to 5..."
	@for id in $(QM9_TASK_ID_RANGE); do \
		echo "Running task with TASK_ID=$$id"; \
		docker run \
			--gpus all \
			-v $(shell pwd)/datasets:/equihgnn/datasets \
			-v $(shell pwd)/logs:/equihgnn/logs \
			-v $(shell pwd)/scripts:/equihgnn/scripts \
			-e COMET_API_KEY=$(COMET_API_KEY) \
			$(IMAGE_NAME) bash scripts/run_qm9_3d.sh $$id; \
	done
	@echo "All training tasks completed."


########## Test OPV ##########
test_opv:
	@echo "Test OPV"
	@for id in $(OPV_TASK_ID_RANGE); do \
		echo "Running task with TASK_ID=$$id"; \
		docker run \
			--gpus all \
			-v $(shell pwd)/datasets:/equihgnn/datasets \
			-v $(shell pwd)/logs:/equihgnn/logs \
			-v $(shell pwd)/scripts:/equihgnn/scripts \
			-v $(shell pwd)/tests:/equihgnn/tests \
			-e COMET_API_KEY=$(COMET_API_KEY) \
			$(IMAGE_NAME) bash tests/run_opv.sh $$id; \
	done
	@echo "Test OPV success"


test_opv_3d:
	@echo "Test OPV-3D"
	@for id in $(OPV_TASK_ID_RANGE); do \
		echo "Running task with TASK_ID=$$id"; \
		docker run \
			--gpus all \
			-v $(shell pwd)/datasets:/equihgnn/datasets \
			-v $(shell pwd)/logs:/equihgnn/logs \
			-v $(shell pwd)/scripts:/equihgnn/scripts \
			-v $(shell pwd)/tests:/equihgnn/tests \
			-e COMET_API_KEY=$(COMET_API_KEY) \
			$(IMAGE_NAME) bash tests/run_opv_3d.sh $$id; \
	done
	@echo "Test OPV-3D success"


########## Test QM9 ##########
test_qm9:
	@echo "Test QM9"
	@for id in $(QM9_TASK_ID_RANGE); do \
		echo "Running task with TASK_ID=$$id"; \
		docker run \
			--gpus all \
			-v $(shell pwd)/datasets:/equihgnn/datasets \
			-v $(shell pwd)/logs:/equihgnn/logs \
			-v $(shell pwd)/scripts:/equihgnn/scripts \
			-v $(shell pwd)/tests:/equihgnn/tests \
			-e COMET_API_KEY=$(COMET_API_KEY) \
			$(IMAGE_NAME) bash tests/run_qm9.sh $$id; \
	done
	@echo "Test OPV success"


test_qm9_3d:
	@echo "Test QM9-3D"
	@for id in $(QM9_TASK_ID_RANGE); do \
		echo "Running task with TASK_ID=$$id"; \
		docker run \
			--gpus all \
			-v $(shell pwd)/datasets:/equihgnn/datasets \
			-v $(shell pwd)/logs:/equihgnn/logs \
			-v $(shell pwd)/scripts:/equihgnn/scripts \
			-v $(shell pwd)/tests:/equihgnn/tests \
			-e COMET_API_KEY=$(COMET_API_KEY) \
			$(IMAGE_NAME) bash tests/run_qm9_3d.sh $$id; \
	done
	@echo "Test OPV-3D success"


########## Test all ##########
test: test_opv test_opv_3d test_qm9 test_qm9_3d
	@echo "All tests completed successfully"