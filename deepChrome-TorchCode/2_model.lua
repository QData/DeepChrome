----------------------------------------------------------------------
-- NN model 
---------------------------------------------------------------------
require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'math'
----------------------------------------------------------------------
print '==>DeepChrome Classification Model' 
print '==> define parameters'

-- Classification problem
noutputs = 2

-- input dimensions
nfeats=5
width = 100
ninputs = nfeats*width

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
nstates = {50,625,125}
filtsize = 10
poolsize = 5
padding = math.floor(filtsize/2)
----------------------------------------------------------------------
print '==> construct model'

if opt.model == 'linear' then

   -- Simple linear model
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,noutputs))

elseif opt.model == 'mlp' then

   -- Simple 2-layer neural network, with tanh hidden units
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,nhiddens))
   model:add(nn.Tanh())
   model:add(nn.Linear(nhiddens,noutputs))

elseif opt.model == 'convnet' then

  
  -- a typical modern convolution network (conv+relu+pool)
  model = nn.Sequential()

  -- stage 1 : filter bank -> squashing -> Max pooling
  model:add(nn.TemporalConvolution(nfeats, nstates[1], filtsize))
  model:add(nn.ReLU())
  model:add(nn.TemporalMaxPooling(poolsize))

  -- stage 2 : standard 2-layer neural network
  model:add(nn.View(math.ceil((width-filtsize)/poolsize)*nstates[1]))
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(math.ceil((width-filtsize)/poolsize)*nstates[1], nstates[2]))
  model:add(nn.ReLU())
  model:add(nn.Linear(nstates[2], nstates[3]))

  model:add(nn.ReLU())
  model:add(nn.Linear(nstates[3], noutputs))

else
   error('unknown -model')

end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------

