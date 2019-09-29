if __name__ == "__main__":
    from gan_task import GANTask
    from config import config
    from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler
    
    scheduler = GPUTaskScheduler(config=config, gpu_task_class=GANTask)
    scheduler.start()