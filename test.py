from redis import Redis

r = Redis(host='localhost', port=6379)
print("Modules:", r.execute_command("MODULE LIST"))
print("FT._LIST:", r.execute_command("FT._LIST"))
