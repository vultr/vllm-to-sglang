pipeline {
    agent any

    parameters {
        string(name: 'BRANCH', defaultValue: 'master', description: 'Git branch to checkout')
        string(name: 'TAG', defaultValue: 'nightly', description: 'Container registry tag to push')
    }

    environment {
        REGISTRY_URL = 'atl.vultrcr.com/vllm/vllm-to-sglang'
        CRED_ID = 'ATL_VCR_VLLM'
        REPO_URL = 'https://sweetapi.com/biondizzle/vllm-to-sglang.git'
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: "${params.BRANCH}", url: "${env.REPO_URL}"
            }
        }
        stage('Build and Push') {
            steps {
                withCredentials([usernamePassword(credentialsId: "${env.CRED_ID}", passwordVariable: 'REG_PASS', usernameVariable: 'REG_USER')]) {
                    sh '''
                    docker login -u "$REG_USER" -p "$REG_PASS" atl.vultrcr.com
                    docker build -t "${REGISTRY_URL}:${TAG}" .
                    docker push "${REGISTRY_URL}:${TAG}"
                    docker logout atl.vultrcr.com
                    '''
                }
            }
        }
    }
}
