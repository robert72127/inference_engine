Needs to support openAI api

frontend: multi user server, use fastapi REST API
async listen for request etc,


also prepare agent for kernels,
try experimenting with automatic improvement loop

for kv cache we want radix

for message passing inter process there is python zmq wrapper,
oh also if one of workers crashes we can restart it,

for message passing inter gpu there is NCCL via torch distributed


Support multi gpu with scheduling
