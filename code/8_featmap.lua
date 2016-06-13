----------------------------------------------------------------------
-- This script generates the feature map 
-- on the test data. Nothing fancy here...
--
-- Ritambhara Singh
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
include('auRoc.lua')
require 'nn'
require 'image'
----------------------------------------------------------------------
print '==> loading data'

testdataset={}

function testdataset:size() return opt.tssize end

for i=1,testdataset:size() do
  local input = testset[i].data;     
  local output = testset[i].label;
  testdataset[i] = {input, output}
end

print '==> defining some tools'

--classes

classes = {'1','2'}

filePath=opt.resultsDir .. opt.dataset

print '==> defining test procedure'

print '==> loading model'
epoch=opt.epoch
local modelname = ('model.' .. epoch .. '.net')
local filename = paths.concat(filePath, modelname)
model = torch.load(filename)

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
   
   for i=model:size(),3,-1 do
       print(model:remove(i))
   end

   print(model)

   X= torch.zeros(91,50)

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
      predict=model:forward(input)
      --maxCol=torch.max(predict,1)
      --print(maxCol)

      --for i = 1,91 do
      	  --for j = 1,50 do
	      --if predict[i][j]>0 then	
	      	 --if (predict[i][j]/maxCol[1][j]) ~= 1 then predict[i][j]=0 else predict[i][j]=1 end
	      --end
      	  --end
      --end
      --print(predict)
      
      X=torch.add(X,predict)
      --print(X)
     
   end
   X:div(testdataset:size())
   --print(X)

   Xt=X:t()
   tablename="table-featmap.txt"
   prob_file=io.open(paths.concat(filePath,tablename),"w")
   for i = 1,50 do
      for j = 1,91 do
      	  prob_file:write(tostring(Xt[i][j])..'\t')
      end
      prob_file:write('\n')
  end

end


