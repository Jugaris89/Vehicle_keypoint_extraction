-- Prepare tensors for saving network output
local validSamples = opt.validIters * opt.validBatch
saved = {idxs = torch.Tensor(validSamples),
         preds = torch.Tensor(validSamples, unpack(ref.predDim))}
if opt.saveInput then saved.input = torch.Tensor(validSamples, unpack(ref.inputDim)) end
if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(validSamples, unpack(ref.outputDim[1])) end

-- function Seq2Seq:double()
-- created by zhaopku to fix the problem of CPU mode
-- https://github.com/macournoyer/neuralconvo/issues/52
-- function Seq2Seq:double()
--  self.encoder:double()
--  self.decoder:double()
--
--  if self.criterion then
--    self.criterion:double()
--  end
--end

-- Main processing step
function step(tag)
    --local avgLoss, avgAcc = 0.0, 0.0
	totalLoss=0
    local output, err, idx
    local param, gradparam = model:getParameters()
    local function evalFn(x) return criterion.output, gradparam end
    sum_per_image = 1
    if tag == 'train' then
        model:training()
        --modelSoftmax:training()
        set = 'train'
    else
        model:evaluate()
        --modelSoftmax:evaluate()
        if tag == 'predict' then
			
            print("==> Generating predictions...")
            local nSamples = dataset:size('test')
            saved = {idxs = torch.Tensor(nSamples),
                     preds = torch.Tensor(nSamples, unpack(ref.predDim))}
            if opt.saveInput then saved.input = torch.Tensor(nSamples, unpack(ref.inputDim)) end
            if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(nSamples, unpack(ref.outputDim[1])) end
            set = 'test'
        else
            set = 'valid'
        end
    end

    local nIters = opt[set .. 'Iters']
	
    for i,sample in loader[set]:run() do
        xlua.progress(i, nIters)
        local input, label, occlusion, indices = unpack(sample)
		--print('OCCLUDED')
		--for i = 1,20 do
		--	print(occlusion[i])
		--	print(label_occlusion[i])
		--end

        if opt.GPU ~= -1 then
            -- Convert to CUDA
            input = applyFn(function (x) return x:cuda() end, input)
            label = applyFn(function (x) return x:cuda() end, label)
	        --label_occlusion = applyFn(function (x) return x:cuda() end, label_occlusion)
            --occlusion = applyFn(function (x) return x:cuda() end, occlusion)
        end

        -- Do a forward pass and calculate loss)
        local output = model:forward(input)
	    --local output2 = modelSoftmax:forward(occlusion)
	    --local errSoftmax = criterion:forward(output2,label_occlusion)
        local err = criterion:forward(output, label)
		totalLoss=totalLoss+err
		avgLoss=totalLoss/nIters
        --avgLoss = avgLoss + err / nIters
        if tag == 'train' then
            -- Training: Do backpropagation and optimization
            model:zeroGradParameters()
            model:backward(input, criterion:backward(output, label))
            optfn(evalFn, param, optimState)
		else
            -- Validation: Get flipped output
            output = applyFn(function (x) return x:clone() end, output)
            local flippedOut = model:forward(flip(input))
            flippedOut = applyFn(function (x) return flip(shuffleLR(x)) end, flippedOut)
            output = applyFn(function (x,y) return x:add(y):div(2) end, output, flippedOut)

            -- Save sample
            local bs = opt[set .. 'Batch']
            local tmpIdx = (i-1) * bs + 1
            local tmpOut = output
            if type(tmpOut) == 'table' then tmpOut = output[#output] end
            if opt.saveInput then saved.input:sub(tmpIdx, tmpIdx+bs-1):copy(input) end
            if opt.saveHeatmaps then saved.heatmaps:sub(tmpIdx, tmpIdx+bs-1):copy(tmpOut) end
            saved.idxs:sub(tmpIdx, tmpIdx+bs-1):copy(indices)
            saved.preds:sub(tmpIdx, tmpIdx+bs-1):copy(postprocess(set,indices,output))
        end

        -- Calculate accuracy
        avgAcc = avgAcc + accuracy(output, label) / nIters
		--print("LOOSSSS")
		--print(avgLoss)
		--print(set)
		--print("END_LOSS")
		--avgLoss=100000*avgLoss
		--avgLoss=math.floor(avgLoss)
		--print(avgLoss)
		-- Send to tensorBoard
		if set == 'train' then
			avgLoss1:add_scalar_value("avgLoss", avgLoss)
			error_keypoint2:add_scalar_value("mean_error_keypoint_train", sum_per_image/(i+1))
		else 
			error_keypoint3:add_scalar_value("mean_error_keypoint_valid", sum_per_image/(i+1))
			if set == 'predict' or set == 'test' then
				avgLoss2:add_scalar_value("avgLoss2", avgLoss)
			else
				avgLoss3:add_scalar_value("avgLoss3", avgLoss)
			end
		end
		
	
			
    end


    -- Print and log some useful metrics
    print(string.format("      %s : Loss: %.7f Acc: %.4f"  % {set, avgLoss, avgAcc}))
    if ref.log[set] then
        table.insert(opt.acc[set], avgAcc)
        ref.log[set]:add{
            ['epoch     '] = string.format("%d" % epoch),
            ['loss      '] = string.format("%.6f" % avgLoss),
            ['acc       '] = string.format("%.4f" % avgAcc),
            ['LR        '] = string.format("%g" % optimState.learningRate)
        }
    end

    if (tag == 'valid' and opt.snapshot ~= 0 and epoch % opt.snapshot == 0) or tag == 'predict' then
        -- Take a snapshot
        model:clearState()
        torch.save(paths.concat(opt.save, 'options.t7'), opt)
        torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
        torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
        local predFilename = 'preds.h5'
        if tag == 'predict' then predFilename = 'final_' .. predFilename end
        local predFile = hdf5.open(paths.concat(opt.save,predFilename),'w')
        for k,v in pairs(saved) do predFile:write(k,v) end
        predFile:close()
    end
end

function train() step('train') end
function valid() step('valid') end
function predict() step('predict') end
