local crayon = require("crayon")

-- A clean server should be running on the test_port with:
-- docker run -it -p 7998:8888 -p 7999:8889 --name crayon_lua_test alband/crayon
local test_port = 7999
local cc = crayon.CrayonClient("localhost", test_port)

-- Check empty
local xp_list = cc:get_experiment_names()
for k,v in pairs(xp_list) do
  error("The server should be empty")
end

require 'paths'
paths.dofile('ref.lua')     -- Parse command line input and do global variable initialization
paths.dofile('model.lua')   -- Read in network model
paths.dofile('train.lua')   -- Load up training/testing functions

-- Set up data loader
torch.setnumthreads(1)
local Dataloader = paths.dofile('util/dataloader.lua')
loader = Dataloader.create(opt, dataset, ref)
	
-- Initialize logs
ref.log = {}
ref.log.train = Logger(paths.concat(opt.save, 'train.log'), opt.continue)
ref.log.valid = Logger(paths.concat(opt.save, 'valid.log'), opt.continue)
avgLoss, avgAcc = 0.0, 0.0
avgLoss1 = cc:create_experiment("avgLoss")
avgLoss2 = cc:create_experiment("avgLoss2")
avgLoss3 = cc:create_experiment("avgLoss3")
error_keypoint = cc:create_experiment("error_keypoint2")
error_keypoint2 = cc:create_experiment("error_keypoint3")
error_keypoint3 = cc:create_experiment("error_keypoint4")
error_oclussion = cc:create_experiment("error_occlusion")
-- Main training loop
for i=1,opt.nEpochs do
    print("==> Starting epoch: " .. epoch .. "/" .. (opt.nEpochs + opt.epochNumber - 1))
    if opt.trainIters > 0 then train() end
    if opt.validIters > 0 then valid() end
    epoch = epoch + 1
    collectgarbage()
end

-- Update reference for last epoch
opt.lastEpoch = epoch - 1

-- Save model
model:clearState()
torch.save(paths.concat(opt.save,'options.t7'), opt)
torch.save(paths.concat(opt.save,'optimState.t7'), optimState)
torch.save(paths.concat(opt.save,'final_model_300518.t7'), model)

-- Generate final predictions on validation set
if opt.finalPredictions then
	ref.log = {}
	loader.test = Dataloader(opt, dataset, ref, 'test')
	predict()
end
