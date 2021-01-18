if __name__ == "__main__":
    from gan_generate_data_task import GANGenerateDataTask
    from config_generate_data import config
    from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler
    
    scheduler = GPUTaskScheduler(
        config=config, gpu_task_class=GANGenerateDataTask)
    scheduler.start()