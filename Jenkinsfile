pipeline {
    agent any

    parameters {
        string(name: 'BRANCH',   defaultValue: 'master',  description: 'Git branch to checkout')
        string(name: 'TAG',      defaultValue: 'nightly', description: 'Base tag; final image tag is ${TAG}-${BACKEND}-${PLATFORM}')
        string(name: 'BACKEND',  defaultValue: '',        description: 'Optional filter: build only this backend (e.g. "sglang"). Empty = all backends found under docker/.')
        string(name: 'PLATFORM', defaultValue: '',        description: 'Optional filter: build only this platform (e.g. "rocm", "cuda"). Empty = all platforms.')
    }

    environment {
        REGISTRY_HOST = 'atl.vultrcr.com'
        REGISTRY_URL  = 'atl.vultrcr.com/vllm/vllm-shim'
        CRED_ID       = 'ATL_VCR_VLLM'
        REPO_URL      = 'https://github.com/vultr/vllm-to-sglang.git'
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: "${params.BRANCH}", url: "${env.REPO_URL}"
            }
        }

        stage('Discover matrix') {
            steps {
                script {
                    def lines = sh(returnStdout: true, script: 'ls docker/*/Dockerfile.* 2>/dev/null || true').trim()
                    def combos = lines ? lines.split('\n').collect { p ->
                        def parts = p.split('/')
                        [backend: parts[1], platform: parts[2] - 'Dockerfile.']
                    } : []
                    if (params.BACKEND?.trim())  { combos = combos.findAll { it.backend  == params.BACKEND.trim()  } }
                    if (params.PLATFORM?.trim()) { combos = combos.findAll { it.platform == params.PLATFORM.trim() } }
                    if (combos.isEmpty()) {
                        error "No Dockerfiles match BACKEND='${params.BACKEND}' PLATFORM='${params.PLATFORM}'. Looked under docker/*/Dockerfile.*"
                    }
                    env.COMBOS = combos.collect { "${it.backend}:${it.platform}" }.join(',')
                    echo "Build matrix: ${env.COMBOS.replace(',', ', ')}"
                }
            }
        }

        stage('Build & Push') {
            steps {
                script {
                    def jobs = env.COMBOS.split(',').collectEntries { combo ->
                        def parts      = combo.tokenize(':')
                        def backend    = parts[0]
                        def platform   = parts[1]
                        def dockerfile = "docker/${backend}/Dockerfile.${platform}"
                        def imageRef   = "${env.REGISTRY_URL}:${params.TAG}-${backend}-${platform}"
                        def cacheRef   = "${env.REGISTRY_URL}:${params.TAG}-${backend}-${platform}-cache"

                        ["${backend}/${platform}": {
                            stage("${backend}/${platform}") {
                                docker.withRegistry("https://${env.REGISTRY_HOST}", env.CRED_ID) {
                                    sh """
                                        docker buildx build \\
                                            --file ${dockerfile} \\
                                            --tag ${imageRef} \\
                                            --cache-from type=registry,ref=${cacheRef} \\
                                            --cache-to   type=registry,ref=${cacheRef},mode=min,ignore-error=true \\
                                            --push \\
                                            .
                                    """
                                }
                            }
                        }]
                    }
                    parallel jobs
                }
            }
        }
    }
}
