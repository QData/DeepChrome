----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data.
--
-- Ritambhara Singh
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
include('auRoc.lua')
----------------------------------------------------------------------
print '==> defining test procedure'

local AUC = auRoc:new()
local AUC_target=1

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testdataset:size() do
      -- disp progress
      xlua.progress(t, testdataset:size())

      -- get new sample
      local input = testdataset[t][1]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = testdataset[t][2]

      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)
      
      if target==2 then AUC_target=-1 else AUC_target=1 end
      --print(AUC_target)
      AUC:add(math.exp(pred[1]), AUC_target)
   end

   -- timing
   time = sys.clock() - time
   time = time / testdataset:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   --print AUC score
   local AUROC = AUC:calculateAuc()
   print(' + AUROC: '..AUROC)
  

   -- update log
   testLogger:add{['AUC Score (test set)'] = AUROC}
   
   --logtestfile=io.open(lognametest,"a")			  
   --logtestfile:write(AUROC .. "\n")
   --logtestfile:close()
   

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
   AUC:zero()
end
