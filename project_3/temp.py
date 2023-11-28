import datetime
import pytz
norway_timezone = pytz.timezone('Europe/Oslo')
start_train_time_norway = datetime.datetime.now(norway_timezone)
print(start_train_time_norway)
