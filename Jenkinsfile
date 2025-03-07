pipeline {
    agent any
    
    environment {
        VENV = 'venv'
        IMAGE_NAME = 'ahmedsoltani/4ds2_mlops'
        TAG = 'latest'
        CONTAINER_NAME = 'mlops_container'
    }
    
    stages {
        stage('Setup Virtual Environment') {
            steps {
                sh 'python3 -m venv ${VENV}'
                sh './${VENV}/bin/pip install --upgrade pip'
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh './${VENV}/bin/pip install -r requirements.txt'
            }
        }
        
        stage('Code Quality Checks') {
            steps {
                sh './${VENV}/bin/black model_pipeline.py main.py'
                sh './${VENV}/bin/pylint --fail-under=5.0 model_pipeline.py main.py'
                sh './${VENV}/bin/bandit -r model_pipeline.py main.py'
            }
        }
        
        stage('Prepare Data') {
            steps {
                sh './${VENV}/bin/python -c "from model_pipeline import prepare_data; prepare_data()"'
            }
        }
        
        stage('Train Model') {
            steps {
                sh './${VENV}/bin/python -c "from model_pipeline import prepare_data, train_model; X_train, y_train, _, _, _, _ = prepare_data(); model = train_model(X_train, y_train)"'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t ${IMAGE_NAME}:${TAG} .'
            }
        }
        
        stage('Push Docker Image') {
            steps {
                withDockerRegistry([credentialsId: 'docker-hub-credentials', url: '']) {
                    sh 'docker push ${IMAGE_NAME}:${TAG}'
                }
            }
        }
        
        stage('Deploy Container') {
            steps {
                sh 'docker stop ${CONTAINER_NAME} || true'
                sh 'docker rm ${CONTAINER_NAME} || true'
                sh 'docker run -d -p 8000:8000 --name ${CONTAINER_NAME} ${IMAGE_NAME}:${TAG}'
            }
        }
    }
    
    post {
        always {
            sh 'rm -rf pycache'
            sh 'rm -rf ${VENV}'
            sh 'rm -f *.pkl'
        }
    }
}