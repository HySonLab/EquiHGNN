IMAGE_NAME = equihgnn

# OPV task:
# molecular: 0-gap, 1-homo, 2-lumo, 3-spectral_overlap
# polymer: 4-homo, 5-lumo, 6-gap, 7-optical_lumo
TASK_ID ?= $(word 2, $(MAKECMDGOALS))
TASK_ID_RANGE = $(shell seq 0 7)

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


train_opv:
	@docker run \
		--gpus all \
		-v $(shell pwd)/datasets:/equihgnn/datasets \
		-v $(shell pwd)/logs:/equihgnn/logs \
		-v $(shell pwd)/scripts:/equihgnn/scripts \
		-e COMET_API_KEY=$COMET_API_KEY \
		$(IMAGE_NAME) bash scripts/run_opv.sh $(TASK_ID)


train_opv_all:
	@echo "Starting training for TASK_ID range 0 to 7..."
	@for id in $(TASK_ID_RANGE); do \
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


train_opv3d:
	@docker run \
		--gpus all \
		-v $(shell pwd)/datasets:/equihgnn/datasets \
		-v $(shell pwd)/logs:/equihgnn/logs \
		-v $(shell pwd)/scripts:/equihgnn/scripts \
		-e COMET_API_KEY=$COMET_API_KEY \
		$(IMAGE_NAME) bash scripts/run_opv_3d.sh $(TASK_ID)


train_opv3d_all:
	@echo "Starting training for TASK_ID range 0 to 7..."
	@for id in $(TASK_ID_RANGE); do \
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


test_opv:
	@echo "Test OPV"
	@docker run \
		--gpus all \
		-v $(shell pwd)/datasets:/equihgnn/datasets \
		-v $(shell pwd)/logs:/equihgnn/logs \
		-v $(shell pwd)/scripts:/equihgnn/scripts \
		-v $(shell pwd)/tests:/equihgnn/tests \
		-e COMET_API_KEY=$COMET_API_KEY \
		$(IMAGE_NAME) bash tests/run_opv.sh 0
	@echo "Test OPV success"


test_opv_3d:
	@echo "Test OPV-3D"
	@docker run \
		--gpus all \
		-v $(shell pwd)/datasets:/equihgnn/datasets \
		-v $(shell pwd)/logs:/equihgnn/logs \
		-v $(shell pwd)/scripts:/equihgnn/scripts \
		-v $(shell pwd)/tests:/equihgnn/tests \
		-e COMET_API_KEY=$COMET_API_KEY \
		$(IMAGE_NAME) bash tests/run_opv_3d.sh 0
	@echo "Test OPV-3D success"

test: test_opv test_opv_3d
	@echo "All tests completed successfully"