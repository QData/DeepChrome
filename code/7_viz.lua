----------------------------------------------------------------------
-- This script performs visualization 
-- on the test data.
-- Original code by Jack Lanchantin (jjl5sw@virginia.edu)
-- Modified by Ritambhara Singh (rs3zz@virginia.edu)
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nn'
dtype='torch.DoubleTensor'
----------------------------------------------------------------------
filePath=opt.resultsDir .. opt.dataset



print '==> loading model'
epoch=opt.epoch
local modelname = ('model.' .. epoch .. '.net')
local filename = paths.concat(filePath, modelname)
model = torch.load(filename)


lambda = 0.009
config = {learningRate=.1,momentum=0.9}

crit = nn.ClassNLLCriterion():float()

-- visualization function
function viz_model()

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)

   for i = 1,model:size() do
       str = tostring(model:get(i))
       if string.match(str, "Dropout") then
       	  model:remove(i)
       end
   end

   model:evaluate()
   model=model:float()
   feval = function(X)
    	 output = model:forward(X)
    	 output = nn.View(2):forward(output)
    	 loss = crit:forward(output, torch.Tensor({1}))
    	 df_do = crit:backward(output, torch.Tensor({1}))
    	 inputGrads = model:backward(X,df_do)
	 
    	 return (loss + lambda*(X:norm())^2), (inputGrads + X*2*lambda)
   end
  
  -- SDG Loop
  X= torch.rand(100,5)
  for i =  1,1000 do
      X,f = optim.rmsprop(feval,X,config)
      
      print(f)
  end
  X:clamp(0,1)

  -- Divide by max to normalize
  max = X:max()
  for i = 1,100 do
      sum = X[i]:sum()
      if sum == 0 then
      	 X[i] = torch.zeros(5)
      else
	for j = 1,5 do
            X[i][j] = X[i][j]/max
      	end
      end
  end


  Xt=X:t()
 
  tablename="table.txt"
  prob_file=io.open(paths.concat(filePath,tablename),"w")
  for i = 1,5 do
      for j = 1,100 do
      	  prob_file:write(tostring(Xt[i][j])..'\t')
      end
      prob_file:write('\n')
  end

end

