prepare:
	mkdir -p ./artifact/

build_db:
	python 1_build_database.py

create_db:
	python 2_create_database.py

validate_db:
	python 3_validate_database.py

embeddings:
	python 4_embedding.py

# train:
# 	python train.py

test:
	python 6_test.py

anomalous_queue:
	python 7_anomalous_queue_construction.py

evaluation:
	python 8_evaluation.py

attack_investigation:
	python 9_attack_investigation.py

preprocess: prepare build_db create_db validate_db embeddings

# deep_graph_learning: train test

anomaly_detection: anomalous_queue evaluation

# pipeline: preprocess deep_graph_learning anomaly_detection attack_investigation

