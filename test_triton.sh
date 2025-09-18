# Server ready?
curl -s localhost:8000/v2/health/ready

# See the (auto-completed) model configs Triton is using:
curl -s localhost:8000/v2/models/lgbm_top25/config | jq .
curl -s localhost:8000/v2/models/lgbm_top50/config | jq .
curl -s localhost:8000/v2/models/lgbm_top100/config | jq .0

# Model stats 
curl -s localhost:8000/v2/models/lgbm_top25/stats | jq .