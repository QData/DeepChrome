----------------------------------------------------------------------
-- This script defines training procedure,
-- irrespective of the model/loss functions chosen.
--
-- It is used to 

--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to: SGD
--
-- Ritambhara Singh
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
include('auRoc.lua')
----------------------------------------------------------------------

print '==> loading data'

traindataset={}
function traindataset:size() return opt.trsize end

for i=1,traindataset:size() do
  local input = trainset[i].data;    
  local output = trainset[i].label;
  traindataset[i] = {input, output}
end

testdataset={}
function testdataset:size() return opt.tssize end

for i=1,testdataset:size() do
  local input = testset[i].data;     
  local output = testset[i].label;
  testdataset[i] = {input, output}
end
------------------------------------------------------------------------------

print '==> defining some tools'

--classes

classes = {'1','2'}


-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
local AUC = auRoc:new()
local AUC_target=1

filePath=opt.resultsDir .. opt.dataset

-- Log results to files
trainLogger = optim.Logger(paths.concat(filePath, 'train.log'))
testLogger = optim.Logger(paths.concat(filePath, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector

if model then
   parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer : SGD'


optimState = {
 	   learningRate = opt.learningRate,	
	   weightDecay = opt.weightDecay,
 	   momentum = opt.momentum,
      	   learningRateDecay = 1e-7
}
optimMethod = optim.sgd

----------------------------------------------------------------------
print '==> defining training procedure'

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- training size
   trsize=traindataset:size()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')

   for t=1,traindataset:size() do
      -- disp progress
      xlua.progress(t, traindataset:size())
      
      local input = traindataset[shuffle[t]][1]
      local target = traindataset[shuffle[t]][2]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      
  
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
   	 -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

	 -- reset gradients
   	 gradParameters:zero()
 
	
       	 -- estimate f
       	 local output = model:forward(input)
       	 f = criterion:forward(output,target)
       	 
       	 -- estimate df/dW
       	 local df_do = criterion:backward(output,target)
       	 model:backward(input, df_do)

	 -- update confusion
         confusion:add(output,target)
	 print(target)
	 
	 if target == 2 then AUC_target=-1 else AUC_target=1 end
	 print(AUC_target)

	 AUC:add(math.exp(output[1]), AUC_target)
	 	 
  	 -- return f and df/dX
       	 return f,gradParameters
      end
      optimMethod(feval, parameters, optimState)
   end
 

   -- time taken
   time = sys.clock() - time
   time = time / traindataset:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   --Calculate and print AUC score
   local AUROC = AUC:calculateAuc()
   print(' + AUROC: '..AUROC)

      -- save/log current net
   local modelname = ('model.' .. epoch .. '.net')
   local filename = paths.concat(filePath, modelname)
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)
  

   trainLogger:add{['AUC Score (train set)'] = AUROC}

   -- next epoch
   confusion:zero()
   AUC:zero()

   epoch = epoch + 1
end

