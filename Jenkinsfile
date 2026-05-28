pipeline {
    agent any

    parameters {
        string(name: 'BRANCH',   defaultValue: 'dev',    description: 'Git branch to checkout')
        string(name: 'TAG',      defaultValue: 'dev',    description: 'Base tag; final image tag is ${TAG}-${BACKEND}-${PLATFORM}')
        string(name: 'BACKEND',  defaultValue: 'sglang', description: 'Optional filter: build only this backend (e.g. "sglang"). Empty = all backends found under docker/.')
        string(name: 'PLATFORM', defaultValue: 'rocm',   description: 'Optional filter: build only this platform (e.g. "rocm", "cuda"). Empty = all platforms.')
        string(name: 'ARCH',     defaultValue: 'amd64',  description: 'Optional CPU architecture for buildx (e.g. "arm64"). Empty = native architecture.')
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
                        def dockerfile   = "docker/${backend}/Dockerfile.${platform}"
                        def archTag      = params.ARCH?.trim() ? "-${params.ARCH.trim()}" : ''
                        def platformFlag = params.ARCH?.trim() ? "--platform linux/${params.ARCH.trim()}" : ''
                        def imageRef     = "${env.REGISTRY_URL}:${params.TAG}-${backend}-${platform}${archTag}"
                        def cacheRef     = "${env.REGISTRY_URL}:${params.TAG}-${backend}-${platform}${archTag}-cache"

                        ["${backend}/${platform}${archTag}": {
                            stage("${backend}/${platform}${archTag}") {
                                docker.withRegistry("https://${env.REGISTRY_HOST}", env.CRED_ID) {
                                    def buildCmd = "docker buildx build --file ${dockerfile} ${platformFlag} --tag ${imageRef} --cache-from type=registry,ref=${cacheRef} --cache-to type=registry,ref=${cacheRef},mode=min,ignore-error=true --push ."
                                    sh buildCmd
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
