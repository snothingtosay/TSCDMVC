import logging, time, sys, os

def logsetting(args):
    if not os.path.exists('./log/'):
        os.mkdir("./log/")
    if not os.path.exists('./log/' + str(args.dataset) + '/'):
        os.mkdir('./log/' + str(args.dataset) + '/')
    if not os.path.exists('./log/' + str(args.dataset) + '/' + 
            'Date=' + time.strftime('%Y-%m-%d', time.localtime(time.time()))):
        os.mkdir('./log/' + str(args.dataset) + '/' + 
            'Date=' + time.strftime('%Y-%m-%d', time.localtime(time.time())))

    path = os.path.join("./log/" + str(args.dataset) + "/" + 
        'Date=' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '/' + 
         time.strftime('%H', time.localtime(time.time())) + '.txt')
    
    if not os.path.exists(path):
        open(path, 'w')
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    return logging










