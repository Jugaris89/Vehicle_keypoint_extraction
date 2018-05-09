local M = {}
Dataset = torch.class('pose.Dataset',M)

function Dataset:__init()
    self.nJoints = 20
    self.accIdxs = {1,2,3,4,5,6,7,8,9,10}
    self.flipRef = {{1,3},   {2,4},   {5,6},
                    {7,8}, {9,19}, {10,20},
                    {11,12}, {13,14}, {15,16}, {17,18}}
    -- Pairs of joints for drawing skeleton
    self.skeletonRef = {{1,2,1},    {2,3,1},    {3,7,1},
                        {4,5,2},    {4,7,2},    {5,6,2},
                        {7,9,0},    {9,10,0},
                        {13,9,3},   {11,12,3},  {12,13,3},
                        {14,9,4},   {14,15,4},  {15,16,4}}

    local annot = {}
    local tags = {'index','center','istrain','imgname'}
    local a = hdf5.open(paths.concat(projectDir,'data/mpii/annot_javi.h5'),'r')
    for _,tag in ipairs(tags) do annot[tag] = a:read(tag):all() end
    a:close()
    annot.index:add(1)
--    annot.person:add(1)
--    annot.part:add(1)

    -- Index reference
    if not opt.idxRef then
        local allIdxs = torch.range(1,annot.index:size(1))
        opt.idxRef = {}
        opt.idxRef.test = allIdxs[annot.istrain:eq(0)]
        opt.idxRef.train = allIdxs[annot.istrain:eq(1)]

        if not opt.randomValid then
            -- Use same validation set as used in our paper (and same as Tompson et al)
--            print(annot.center)
            tmpAnnot = annot.index:cat(annot.imgname, 2):long()
            tmpAnnot:add(-1)

            local validAnnot = hdf5.open(paths.concat(projectDir, 'data/mpii/annot/valid.h5'),'r')
            local tmpValid = validAnnot:read('index'):all():cat(validAnnot:read('imgname'):all(),2):long()
            print('DEBUG1')
            print(tmpValid)
            opt.idxRef.valid = torch.zeros(tmpValid:size(1))
	    opt.nValidImgs = opt.idxRef.valid:size(1)
            print('DEBUG2')
	    print(opt.nValidImgs)
            print(opt.idxRef.train:size(1))
	    opt.idxRef.train = torch.zeros(opt.idxRef.train:size(1) - opt.nValidImgs)

            -- Loop through to get proper index values
            local validCount = 1
            local trainCount = 1
            for i = 1,annot.index:size(1) do
                if validCount <= tmpValid:size(1) and tmpAnnot[i]:equal(tmpValid[validCount]) then
                    opt.idxRef.valid[validCount] = i
                    validCount = validCount + 1
                elseif annot.istrain[i] == 1 then
                    --print(opt.idxRef.train)
		    opt.idxRef.train[trainCount] = i
                    trainCount = trainCount + 1
                end
            end
        else
            -- Set up random training/validation split
            local perm = torch.randperm(opt.idxRef.train:size(1)):long()
            opt.idxRef.valid = opt.idxRef.train:index(1, perm:sub(1,opt.nValidImgs))
            opt.idxRef.train = opt.idxRef.train:index(1, perm:sub(opt.nValidImgs+1,-1))
        end

        torch.save(opt.save .. '/options.t7', opt)
    end

    self.annot = annot
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel(),
                     test=opt.idxRef.test:numel()}

    -- For final predictions
    opt.testIters = self.nsamples.test
    opt.testBatch = 1
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)
    print("JAAVI")
    print(ffi.string(self.annot.imgname[idx]:char():data()))
    return paths.concat(opt.dataDir,'images',ffi.string(self.annot.imgname[idx]:char():data()))
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx)
    local pts = self.annot.center[idx]:clone()
    local c = self.annot.center[idx]:clone()
    --print(self.annot.scale)
    local s = 1 --self.annot.scale[idx]
    --print(s)
    -- Small adjustment so cropping is less likely to take feet out
    c[2] = c[2] + 15 * s
    s = s * 1.25
--    return c
    return pts, c, s
end

function Dataset:normalize(idx)
    return self.annot.normalize[idx]
end

return M.Dataset

