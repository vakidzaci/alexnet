version: '3.1'
services:
  rabbitmq:
    image: rabbitmq:3-management-alpine
    network_mode: bridge
    hostname: rabbit-container
    volumes:
      - rabbitmq-data:/var/lib/rabbitmq
    ports:
      - 15672:15672
      - 5672:5672
    healthcheck:
      test: ['CMD', 'rabbitmqctl', 'status']
      interval: 30s
      timeout: 15s
      retries: 3
    restart: unless-stopped
  redis:
    image: "redis:alpine"
    network_mode: bridge
    container_name : "redis"
    ports:
      - "6379:6379"
  ocr_worker:
    image : ocr_service:openvinov0.4
    network_mode: bridge
    cpus: 2
    mem_limit: 4g
    volumes:
      - ./input:/app/input
      - ./tmp:/app/tmp
      - ./results:/app/results
      - ./flask_celery:/app/flask_celery
    command: celery -A flask_celery.tasks worker -l DEBUG -Q ocr_queue -c 1 -P threads -n worker@%n --prefetch-multiplier=1 --max-tasks-per-child=10 --time-limit=1000 --soft-time-limit=1000
    links:
      - rabbitmq
      - redis
    depends_on:
      - rabbitmq
      - redis
  read_page_worker:
    image: ocr_service:openvinov0.4
    network_mode: bridge
    cpus: 16
    mem_limit: 32g
    volumes:
      - ./input:/app/input
      - ./tmp:/app/tmp
      - ./results:/app/results
      - ./flask_celery:/app/flask_celery
    command: celery -A flask_celery.tasks worker -l DEBUG -Q read_page_queue -c 1 -P threads -n worker@%n --prefetch-multiplier=1 --max-tasks-per-child=1 --time-limit=1000 --soft-time-limit=1000
    links:
      - rabbitmq
      - redis
    depends_on:
      - rabbitmq
      - redis
  collect_results_worker:
    image: ocr_service:openvinov0.4
    container_name : "collect_results_worker"
    network_mode: bridge
    cpus: 1
    mem_limit: 2g
    volumes:
      - ./input:/app/input
      - ./tmp:/app/tmp
      - ./results:/app/results
      - ./flask_celery:/app/flask_celery
    command : celery -A flask_celery.tasks worker -l DEBUG -Q collect_results_queue -c 1 -P threads -n worker@%n --prefetch-multiplier=1 --max-tasks-per-child=10 --time-limit=1000 --soft-time-limit=1000
    links:
      - rabbitmq
      - redis
    depends_on:
      - rabbitmq
      - redis
  remove_results_worker:
    image: ocr_service:openvinov0.4
    container_name: "remove_results_worker"
    network_mode: bridge
    cpus: 1
    mem_limit: 2g
    volumes:
      - ./input:/app/input
      - ./tmp:/app/tmp
      - ./results:/app/results
      - ./flask_celery:/app/flask_celery
    command: celery -A flask_celery.tasks worker -l DEBUG -Q remove_dir_queue -c 1 -P threads -n worker@%n --prefetch-multiplier=1 --max-tasks-per-child=10 --time-limit=1000 --soft-time-limit=1000
    links:
      - rabbitmq
      - redis
    depends_on:
      - rabbitmq
      - redis
  flower:
    image: ocr_service:openvinov0.4
    container_name: "flower"
    network_mode: bridge
    ports:
      - "5555:5555"
    volumes:
      - ./input:/app/input
      - ./tmp:/app/tmp
      - ./results:/app/results
      - ./flask_celery:/app/flask_celery
    command: celery -A flask_celery.tasks flower  --port=5555
    links:
      - rabbitmq
      - redis
    depends_on:
      - rabbitmq
      - redis
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
    links:
      - rabbitmq
      - redis
      - flower
    depends_on:
      - rabbitmq
      - redis
      - flower
  app:
    image: ocr_service:openvinov0.4
    network_mode: bridge
    volumes:
      - ./input:/app/input
      - ./tmp:/app/tmp
      - ./results:/app/results
      - ./flask_celery:/app/flask_celery
    command: python3.8 run.py
    ports:
      - '5000:5000'
    links:
      - rabbitmq
      - redis
    depends_on:
      - rabbitmq
      - redis

    restart: unless-stopped
volumes:
  rabbitmq-data:
