˛­
Ć
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018˝

SGD/dense_10/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/dense_10/bias/momentum

.SGD/dense_10/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_10/bias/momentum*
_output_shapes	
:*
dtype0

SGD/dense_10/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameSGD/dense_10/kernel/momentum

0SGD/dense_10/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_10/kernel/momentum* 
_output_shapes
:
*
dtype0
Š
(SGD/batch_normalization_42/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(SGD/batch_normalization_42/beta/momentum
˘
<SGD/batch_normalization_42/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_42/beta/momentum*
_output_shapes	
:*
dtype0
Ť
)SGD/batch_normalization_42/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)SGD/batch_normalization_42/gamma/momentum
¤
=SGD/batch_normalization_42/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_42/gamma/momentum*
_output_shapes	
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:*
dtype0
|
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
*
dtype0
Ľ
&batch_normalization_42/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_42/moving_variance

:batch_normalization_42/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_42/moving_variance*
_output_shapes	
:*
dtype0

"batch_normalization_42/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_42/moving_mean

6batch_normalization_42/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_42/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization_42/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_42/beta

/batch_normalization_42/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_42/beta*
_output_shapes	
:*
dtype0

batch_normalization_42/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_42/gamma

0batch_normalization_42/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_42/gamma*
_output_shapes	
:*
dtype0

NoOpNoOp
¨'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ă&
valueŮ&BÖ& BĎ&
§
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
Ő
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axis
	gamma
beta
moving_mean
moving_variance*
Ś
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
.
0
1
2
3
$4
%5*
 
0
1
$2
%3*
* 
°
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
+trace_0
,trace_1
-trace_2
.trace_3* 
6
/trace_0
0trace_1
1trace_2
2trace_3* 
* 
z
3iter
	4decay
5learning_rate
6momentummomentumZmomentum[$momentum\%momentum]*

7serving_default* 
* 
* 
* 

8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

=trace_0* 

>trace_0* 
 
0
1
2
3*

0
1*
* 

?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Dtrace_0
Etrace_1* 

Ftrace_0
Gtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_42/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_42/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_42/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_42/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 

Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

Mtrace_0* 

Ntrace_0* 
_Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_10/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1
2*

O0
P1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
KE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
Q	variables
R	keras_api
	Stotal
	Tcount*
H
U	variables
V	keras_api
	Wtotal
	Xcount
Y
_fn_kwargs*

S0
T1*

Q	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

W0
X1*

U	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

VARIABLE_VALUE)SGD/batch_normalization_42/gamma/momentumXlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE(SGD/batch_normalization_42/beta/momentumWlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUESGD/dense_10/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUESGD/dense_10/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
Ś
1serving_default_global_average_pooling2d_11_inputPlaceholder*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*%
shape:˙˙˙˙˙˙˙˙˙

StatefulPartitionedCallStatefulPartitionedCall1serving_default_global_average_pooling2d_11_input&batch_normalization_42/moving_variancebatch_normalization_42/gamma"batch_normalization_42/moving_meanbatch_normalization_42/betadense_10/kerneldense_10/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_956009
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ą
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0batch_normalization_42/gamma/Read/ReadVariableOp/batch_normalization_42/beta/Read/ReadVariableOp6batch_normalization_42/moving_mean/Read/ReadVariableOp:batch_normalization_42/moving_variance/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp=SGD/batch_normalization_42/gamma/momentum/Read/ReadVariableOp<SGD/batch_normalization_42/beta/momentum/Read/ReadVariableOp0SGD/dense_10/kernel/momentum/Read/ReadVariableOp.SGD/dense_10/bias/momentum/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_956303
´
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_42/gammabatch_normalization_42/beta"batch_normalization_42/moving_mean&batch_normalization_42/moving_variancedense_10/kerneldense_10/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotal_1count_1totalcount)SGD/batch_normalization_42/gamma/momentum(SGD/batch_normalization_42/beta/momentumSGD/dense_10/kernel/momentumSGD/dense_10/bias/momentum*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_956367ßż
á
ľ
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_956172

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity˘batchnorm/ReadVariableOp˘batchnorm/ReadVariableOp_1˘batchnorm/ReadVariableOp_2˘batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ş
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¸
s
W__inference_global_average_pooling2d_11_layer_call_and_return_conditional_losses_955733

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ź

ř
D__inference_dense_10_layer_call_and_return_conditional_losses_955846

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ź

ř
D__inference_dense_10_layer_call_and_return_conditional_losses_956226

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
K
Ą
"__inference__traced_restore_956367
file_prefix<
-assignvariableop_batch_normalization_42_gamma:	=
.assignvariableop_1_batch_normalization_42_beta:	D
5assignvariableop_2_batch_normalization_42_moving_mean:	H
9assignvariableop_3_batch_normalization_42_moving_variance:	6
"assignvariableop_4_dense_10_kernel:
/
 assignvariableop_5_dense_10_bias:	%
assignvariableop_6_sgd_iter:	 &
assignvariableop_7_sgd_decay: .
$assignvariableop_8_sgd_learning_rate: )
assignvariableop_9_sgd_momentum: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: #
assignvariableop_12_total: #
assignvariableop_13_count: L
=assignvariableop_14_sgd_batch_normalization_42_gamma_momentum:	K
<assignvariableop_15_sgd_batch_normalization_42_beta_momentum:	D
0assignvariableop_16_sgd_dense_10_kernel_momentum:
=
.assignvariableop_17_sgd_dense_10_bias_momentum:	
identity_19˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9ß	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*	
valueűBřB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ý
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp-assignvariableop_batch_normalization_42_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp.assignvariableop_1_batch_normalization_42_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_2AssignVariableOp5assignvariableop_2_batch_normalization_42_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_3AssignVariableOp9assignvariableop_3_batch_normalization_42_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_10_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_10_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_sgd_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp$assignvariableop_8_sgd_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ž
AssignVariableOp_14AssignVariableOp=assignvariableop_14_sgd_batch_normalization_42_gamma_momentumIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_15AssignVariableOp<assignvariableop_15_sgd_batch_normalization_42_beta_momentumIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ą
AssignVariableOp_16AssignVariableOp0assignvariableop_16_sgd_dense_10_kernel_momentumIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp.assignvariableop_17_sgd_dense_10_bias_momentumIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ű
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: Č
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ö	
Ą
.__inference_sequential_12_layer_call_fn_955948%
!global_average_pooling2d_11_input
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:

	unknown_4:	
identity˘StatefulPartitionedCallŹ
StatefulPartitionedCallStatefulPartitionedCall!global_average_pooling2d_11_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_955916p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:s o
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
;
_user_specified_name#!global_average_pooling2d_11_input
Ř	
Ą
.__inference_sequential_12_layer_call_fn_955868%
!global_average_pooling2d_11_input
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:

	unknown_4:	
identity˘StatefulPartitionedCallŽ
StatefulPartitionedCallStatefulPartitionedCall!global_average_pooling2d_11_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_955853p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:s o
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
;
_user_specified_name#!global_average_pooling2d_11_input
¸
s
W__inference_global_average_pooling2d_11_layer_call_and_return_conditional_losses_956126

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°%
ď
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_955807

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity˘AssignMovingAvg˘AssignMovingAvg/ReadVariableOp˘AssignMovingAvg_1˘ AssignMovingAvg_1/ReadVariableOp˘batchnorm/ReadVariableOp˘batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:Ź
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ę
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
É

)__inference_dense_10_layer_call_fn_956215

inputs
unknown:

	unknown_0:	
identity˘StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_955846p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°
Ö
7__inference_batch_normalization_42_layer_call_fn_956152

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_955807p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ś	

$__inference_signature_wrapper_956009%
!global_average_pooling2d_11_input
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:

	unknown_4:	
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall!global_average_pooling2d_11_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_955723p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:s o
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
;
_user_specified_name#!global_average_pooling2d_11_input
	

.__inference_sequential_12_layer_call_fn_956026

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:

	unknown_4:	
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_955853p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	

.__inference_sequential_12_layer_call_fn_956043

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:

	unknown_4:	
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_955916p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

X
<__inference_global_average_pooling2d_11_layer_call_fn_956120

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_global_average_pooling2d_11_layer_call_and_return_conditional_losses_955733i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Đ
Ô
I__inference_sequential_12_layer_call_and_return_conditional_losses_955967%
!global_average_pooling2d_11_input,
batch_normalization_42_955952:	,
batch_normalization_42_955954:	,
batch_normalization_42_955956:	,
batch_normalization_42_955958:	#
dense_10_955961:

dense_10_955963:	
identity˘.batch_normalization_42/StatefulPartitionedCall˘ dense_10/StatefulPartitionedCallú
+global_average_pooling2d_11/PartitionedCallPartitionedCall!global_average_pooling2d_11_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_global_average_pooling2d_11_layer_call_and_return_conditional_losses_955733
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_11/PartitionedCall:output:0batch_normalization_42_955952batch_normalization_42_955954batch_normalization_42_955956batch_normalization_42_955958*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_955760˘
 dense_10/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0dense_10_955961dense_10_955963*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_955846y
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp/^batch_normalization_42/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙: : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall:s o
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
;
_user_specified_name#!global_average_pooling2d_11_input
ý
š
I__inference_sequential_12_layer_call_and_return_conditional_losses_955916

inputs,
batch_normalization_42_955901:	,
batch_normalization_42_955903:	,
batch_normalization_42_955905:	,
batch_normalization_42_955907:	#
dense_10_955910:

dense_10_955912:	
identity˘.batch_normalization_42/StatefulPartitionedCall˘ dense_10/StatefulPartitionedCallß
+global_average_pooling2d_11/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_global_average_pooling2d_11_layer_call_and_return_conditional_losses_955733
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_11/PartitionedCall:output:0batch_normalization_42_955901batch_normalization_42_955903batch_normalization_42_955905batch_normalization_42_955907*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_955807˘
 dense_10/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0dense_10_955910dense_10_955912*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_955846y
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp/^batch_normalization_42/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙: : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˛
Ö
7__inference_batch_normalization_42_layer_call_fn_956139

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_955760p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˙-
Ś
__inference__traced_save_956303
file_prefix;
7savev2_batch_normalization_42_gamma_read_readvariableop:
6savev2_batch_normalization_42_beta_read_readvariableopA
=savev2_batch_normalization_42_moving_mean_read_readvariableopE
Asavev2_batch_normalization_42_moving_variance_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopH
Dsavev2_sgd_batch_normalization_42_gamma_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_42_beta_momentum_read_readvariableop;
7savev2_sgd_dense_10_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_10_bias_momentum_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ü	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*	
valueűBřB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ´
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_batch_normalization_42_gamma_read_readvariableop6savev2_batch_normalization_42_beta_read_readvariableop=savev2_batch_normalization_42_moving_mean_read_readvariableopAsavev2_batch_normalization_42_moving_variance_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopDsavev2_sgd_batch_normalization_42_gamma_momentum_read_readvariableopCsavev2_sgd_batch_normalization_42_beta_momentum_read_readvariableop7savev2_sgd_dense_10_kernel_momentum_read_readvariableop5savev2_sgd_dense_10_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*w
_input_shapesf
d: :::::
:: : : : : : : : :::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: 
˙
š
I__inference_sequential_12_layer_call_and_return_conditional_losses_955853

inputs,
batch_normalization_42_955826:	,
batch_normalization_42_955828:	,
batch_normalization_42_955830:	,
batch_normalization_42_955832:	#
dense_10_955847:

dense_10_955849:	
identity˘.batch_normalization_42/StatefulPartitionedCall˘ dense_10/StatefulPartitionedCallß
+global_average_pooling2d_11/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_global_average_pooling2d_11_layer_call_and_return_conditional_losses_955733
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_11/PartitionedCall:output:0batch_normalization_42_955826batch_normalization_42_955828batch_normalization_42_955830batch_normalization_42_955832*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_955760˘
 dense_10/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0dense_10_955847dense_10_955849*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_955846y
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp/^batch_normalization_42/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙: : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
á
ľ
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_955760

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity˘batchnorm/ReadVariableOp˘batchnorm/ReadVariableOp_1˘batchnorm/ReadVariableOp_2˘batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ş
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°%
ď
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_956206

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity˘AssignMovingAvg˘AssignMovingAvg/ReadVariableOp˘AssignMovingAvg_1˘ AssignMovingAvg_1/ReadVariableOp˘batchnorm/ReadVariableOp˘batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:Ź
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ę
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
őA

I__inference_sequential_12_layer_call_and_return_conditional_losses_956115

inputsM
>batch_normalization_42_assignmovingavg_readvariableop_resource:	O
@batch_normalization_42_assignmovingavg_1_readvariableop_resource:	K
<batch_normalization_42_batchnorm_mul_readvariableop_resource:	G
8batch_normalization_42_batchnorm_readvariableop_resource:	;
'dense_10_matmul_readvariableop_resource:
7
(dense_10_biasadd_readvariableop_resource:	
identity˘&batch_normalization_42/AssignMovingAvg˘5batch_normalization_42/AssignMovingAvg/ReadVariableOp˘(batch_normalization_42/AssignMovingAvg_1˘7batch_normalization_42/AssignMovingAvg_1/ReadVariableOp˘/batch_normalization_42/batchnorm/ReadVariableOp˘3batch_normalization_42/batchnorm/mul/ReadVariableOp˘dense_10/BiasAdd/ReadVariableOp˘dense_10/MatMul/ReadVariableOp
2global_average_pooling2d_11/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
 global_average_pooling2d_11/MeanMeaninputs;global_average_pooling2d_11/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
5batch_normalization_42/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ń
#batch_normalization_42/moments/meanMean)global_average_pooling2d_11/Mean:output:0>batch_normalization_42/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
+batch_normalization_42/moments/StopGradientStopGradient,batch_normalization_42/moments/mean:output:0*
T0*
_output_shapes
:	Ů
0batch_normalization_42/moments/SquaredDifferenceSquaredDifference)global_average_pooling2d_11/Mean:output:04batch_normalization_42/moments/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
9batch_normalization_42/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ä
'batch_normalization_42/moments/varianceMean4batch_normalization_42/moments/SquaredDifference:z:0Bbatch_normalization_42/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
&batch_normalization_42/moments/SqueezeSqueeze,batch_normalization_42/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 ˘
(batch_normalization_42/moments/Squeeze_1Squeeze0batch_normalization_42/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 q
,batch_normalization_42/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<ą
5batch_normalization_42/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_42_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0Ç
*batch_normalization_42/AssignMovingAvg/subSub=batch_normalization_42/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_42/moments/Squeeze:output:0*
T0*
_output_shapes	
:ž
*batch_normalization_42/AssignMovingAvg/mulMul.batch_normalization_42/AssignMovingAvg/sub:z:05batch_normalization_42/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:
&batch_normalization_42/AssignMovingAvgAssignSubVariableOp>batch_normalization_42_assignmovingavg_readvariableop_resource.batch_normalization_42/AssignMovingAvg/mul:z:06^batch_normalization_42/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_42/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<ľ
7batch_normalization_42/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_42_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0Í
,batch_normalization_42/AssignMovingAvg_1/subSub?batch_normalization_42/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_42/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ä
,batch_normalization_42/AssignMovingAvg_1/mulMul0batch_normalization_42/AssignMovingAvg_1/sub:z:07batch_normalization_42/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:
(batch_normalization_42/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_42_assignmovingavg_1_readvariableop_resource0batch_normalization_42/AssignMovingAvg_1/mul:z:08^batch_normalization_42/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_42/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ˇ
$batch_normalization_42/batchnorm/addAddV21batch_normalization_42/moments/Squeeze_1:output:0/batch_normalization_42/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
&batch_normalization_42/batchnorm/RsqrtRsqrt(batch_normalization_42/batchnorm/add:z:0*
T0*
_output_shapes	
:­
3batch_normalization_42/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_42_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0ş
$batch_normalization_42/batchnorm/mulMul*batch_normalization_42/batchnorm/Rsqrt:y:0;batch_normalization_42/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ľ
&batch_normalization_42/batchnorm/mul_1Mul)global_average_pooling2d_11/Mean:output:0(batch_normalization_42/batchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ž
&batch_normalization_42/batchnorm/mul_2Mul/batch_normalization_42/moments/Squeeze:output:0(batch_normalization_42/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ľ
/batch_normalization_42/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_42_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0ś
$batch_normalization_42/batchnorm/subSub7batch_normalization_42/batchnorm/ReadVariableOp:value:0*batch_normalization_42/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¸
&batch_normalization_42/batchnorm/add_1AddV2*batch_normalization_42/batchnorm/mul_1:z:0(batch_normalization_42/batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
dense_10/MatMulMatMul*batch_normalization_42/batchnorm/add_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
IdentityIdentitydense_10/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ˇ
NoOpNoOp'^batch_normalization_42/AssignMovingAvg6^batch_normalization_42/AssignMovingAvg/ReadVariableOp)^batch_normalization_42/AssignMovingAvg_18^batch_normalization_42/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_42/batchnorm/ReadVariableOp4^batch_normalization_42/batchnorm/mul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙: : : : : : 2P
&batch_normalization_42/AssignMovingAvg&batch_normalization_42/AssignMovingAvg2n
5batch_normalization_42/AssignMovingAvg/ReadVariableOp5batch_normalization_42/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_42/AssignMovingAvg_1(batch_normalization_42/AssignMovingAvg_12r
7batch_normalization_42/AssignMovingAvg_1/ReadVariableOp7batch_normalization_42/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_42/batchnorm/ReadVariableOp/batch_normalization_42/batchnorm/ReadVariableOp2j
3batch_normalization_42/batchnorm/mul/ReadVariableOp3batch_normalization_42/batchnorm/mul/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
0
¸
!__inference__wrapped_model_955723%
!global_average_pooling2d_11_inputU
Fsequential_12_batch_normalization_42_batchnorm_readvariableop_resource:	Y
Jsequential_12_batch_normalization_42_batchnorm_mul_readvariableop_resource:	W
Hsequential_12_batch_normalization_42_batchnorm_readvariableop_1_resource:	W
Hsequential_12_batch_normalization_42_batchnorm_readvariableop_2_resource:	I
5sequential_12_dense_10_matmul_readvariableop_resource:
E
6sequential_12_dense_10_biasadd_readvariableop_resource:	
identity˘=sequential_12/batch_normalization_42/batchnorm/ReadVariableOp˘?sequential_12/batch_normalization_42/batchnorm/ReadVariableOp_1˘?sequential_12/batch_normalization_42/batchnorm/ReadVariableOp_2˘Asequential_12/batch_normalization_42/batchnorm/mul/ReadVariableOp˘-sequential_12/dense_10/BiasAdd/ReadVariableOp˘,sequential_12/dense_10/MatMul/ReadVariableOp
@sequential_12/global_average_pooling2d_11/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ×
.sequential_12/global_average_pooling2d_11/MeanMean!global_average_pooling2d_11_inputIsequential_12/global_average_pooling2d_11/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Á
=sequential_12/batch_normalization_42/batchnorm/ReadVariableOpReadVariableOpFsequential_12_batch_normalization_42_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0y
4sequential_12/batch_normalization_42/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ç
2sequential_12/batch_normalization_42/batchnorm/addAddV2Esequential_12/batch_normalization_42/batchnorm/ReadVariableOp:value:0=sequential_12/batch_normalization_42/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
4sequential_12/batch_normalization_42/batchnorm/RsqrtRsqrt6sequential_12/batch_normalization_42/batchnorm/add:z:0*
T0*
_output_shapes	
:É
Asequential_12/batch_normalization_42/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_12_batch_normalization_42_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0ä
2sequential_12/batch_normalization_42/batchnorm/mulMul8sequential_12/batch_normalization_42/batchnorm/Rsqrt:y:0Isequential_12/batch_normalization_42/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ß
4sequential_12/batch_normalization_42/batchnorm/mul_1Mul7sequential_12/global_average_pooling2d_11/Mean:output:06sequential_12/batch_normalization_42/batchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ĺ
?sequential_12/batch_normalization_42/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_12_batch_normalization_42_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0â
4sequential_12/batch_normalization_42/batchnorm/mul_2MulGsequential_12/batch_normalization_42/batchnorm/ReadVariableOp_1:value:06sequential_12/batch_normalization_42/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ĺ
?sequential_12/batch_normalization_42/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_12_batch_normalization_42_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0â
2sequential_12/batch_normalization_42/batchnorm/subSubGsequential_12/batch_normalization_42/batchnorm/ReadVariableOp_2:value:08sequential_12/batch_normalization_42/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:â
4sequential_12/batch_normalization_42/batchnorm/add_1AddV28sequential_12/batch_normalization_42/batchnorm/mul_1:z:06sequential_12/batch_normalization_42/batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
,sequential_12/dense_10/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ę
sequential_12/dense_10/MatMulMatMul8sequential_12/batch_normalization_42/batchnorm/add_1:z:04sequential_12/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
-sequential_12/dense_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ź
sequential_12/dense_10/BiasAddBiasAdd'sequential_12/dense_10/MatMul:product:05sequential_12/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
sequential_12/dense_10/SoftmaxSoftmax'sequential_12/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
IdentityIdentity(sequential_12/dense_10/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙­
NoOpNoOp>^sequential_12/batch_normalization_42/batchnorm/ReadVariableOp@^sequential_12/batch_normalization_42/batchnorm/ReadVariableOp_1@^sequential_12/batch_normalization_42/batchnorm/ReadVariableOp_2B^sequential_12/batch_normalization_42/batchnorm/mul/ReadVariableOp.^sequential_12/dense_10/BiasAdd/ReadVariableOp-^sequential_12/dense_10/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙: : : : : : 2~
=sequential_12/batch_normalization_42/batchnorm/ReadVariableOp=sequential_12/batch_normalization_42/batchnorm/ReadVariableOp2
?sequential_12/batch_normalization_42/batchnorm/ReadVariableOp_1?sequential_12/batch_normalization_42/batchnorm/ReadVariableOp_12
?sequential_12/batch_normalization_42/batchnorm/ReadVariableOp_2?sequential_12/batch_normalization_42/batchnorm/ReadVariableOp_22
Asequential_12/batch_normalization_42/batchnorm/mul/ReadVariableOpAsequential_12/batch_normalization_42/batchnorm/mul/ReadVariableOp2^
-sequential_12/dense_10/BiasAdd/ReadVariableOp-sequential_12/dense_10/BiasAdd/ReadVariableOp2\
,sequential_12/dense_10/MatMul/ReadVariableOp,sequential_12/dense_10/MatMul/ReadVariableOp:s o
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
;
_user_specified_name#!global_average_pooling2d_11_input
Ń'

I__inference_sequential_12_layer_call_and_return_conditional_losses_956072

inputsG
8batch_normalization_42_batchnorm_readvariableop_resource:	K
<batch_normalization_42_batchnorm_mul_readvariableop_resource:	I
:batch_normalization_42_batchnorm_readvariableop_1_resource:	I
:batch_normalization_42_batchnorm_readvariableop_2_resource:	;
'dense_10_matmul_readvariableop_resource:
7
(dense_10_biasadd_readvariableop_resource:	
identity˘/batch_normalization_42/batchnorm/ReadVariableOp˘1batch_normalization_42/batchnorm/ReadVariableOp_1˘1batch_normalization_42/batchnorm/ReadVariableOp_2˘3batch_normalization_42/batchnorm/mul/ReadVariableOp˘dense_10/BiasAdd/ReadVariableOp˘dense_10/MatMul/ReadVariableOp
2global_average_pooling2d_11/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
 global_average_pooling2d_11/MeanMeaninputs;global_average_pooling2d_11/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ľ
/batch_normalization_42/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_42_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0k
&batch_normalization_42/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:˝
$batch_normalization_42/batchnorm/addAddV27batch_normalization_42/batchnorm/ReadVariableOp:value:0/batch_normalization_42/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
&batch_normalization_42/batchnorm/RsqrtRsqrt(batch_normalization_42/batchnorm/add:z:0*
T0*
_output_shapes	
:­
3batch_normalization_42/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_42_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0ş
$batch_normalization_42/batchnorm/mulMul*batch_normalization_42/batchnorm/Rsqrt:y:0;batch_normalization_42/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ľ
&batch_normalization_42/batchnorm/mul_1Mul)global_average_pooling2d_11/Mean:output:0(batch_normalization_42/batchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
1batch_normalization_42/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_42_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0¸
&batch_normalization_42/batchnorm/mul_2Mul9batch_normalization_42/batchnorm/ReadVariableOp_1:value:0(batch_normalization_42/batchnorm/mul:z:0*
T0*
_output_shapes	
:Š
1batch_normalization_42/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_42_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0¸
$batch_normalization_42/batchnorm/subSub9batch_normalization_42/batchnorm/ReadVariableOp_2:value:0*batch_normalization_42/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¸
&batch_normalization_42/batchnorm/add_1AddV2*batch_normalization_42/batchnorm/mul_1:z:0(batch_normalization_42/batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
dense_10/MatMulMatMul*batch_normalization_42/batchnorm/add_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
IdentityIdentitydense_10/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ů
NoOpNoOp0^batch_normalization_42/batchnorm/ReadVariableOp2^batch_normalization_42/batchnorm/ReadVariableOp_12^batch_normalization_42/batchnorm/ReadVariableOp_24^batch_normalization_42/batchnorm/mul/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙: : : : : : 2b
/batch_normalization_42/batchnorm/ReadVariableOp/batch_normalization_42/batchnorm/ReadVariableOp2f
1batch_normalization_42/batchnorm/ReadVariableOp_11batch_normalization_42/batchnorm/ReadVariableOp_12f
1batch_normalization_42/batchnorm/ReadVariableOp_21batch_normalization_42/batchnorm/ReadVariableOp_22j
3batch_normalization_42/batchnorm/mul/ReadVariableOp3batch_normalization_42/batchnorm/mul/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Î
Ô
I__inference_sequential_12_layer_call_and_return_conditional_losses_955986%
!global_average_pooling2d_11_input,
batch_normalization_42_955971:	,
batch_normalization_42_955973:	,
batch_normalization_42_955975:	,
batch_normalization_42_955977:	#
dense_10_955980:

dense_10_955982:	
identity˘.batch_normalization_42/StatefulPartitionedCall˘ dense_10/StatefulPartitionedCallú
+global_average_pooling2d_11/PartitionedCallPartitionedCall!global_average_pooling2d_11_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_global_average_pooling2d_11_layer_call_and_return_conditional_losses_955733
.batch_normalization_42/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_11/PartitionedCall:output:0batch_normalization_42_955971batch_normalization_42_955973batch_normalization_42_955975batch_normalization_42_955977*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_955807˘
 dense_10/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_42/StatefulPartitionedCall:output:0dense_10_955980dense_10_955982*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_955846y
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp/^batch_normalization_42/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙: : : : : : 2`
.batch_normalization_42/StatefulPartitionedCall.batch_normalization_42/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall:s o
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
;
_user_specified_name#!global_average_pooling2d_11_input"żL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*é
serving_defaultŐ
x
!global_average_pooling2d_11_inputS
3serving_default_global_average_pooling2d_11_input:0˙˙˙˙˙˙˙˙˙=
dense_101
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:{
Á
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
Ľ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
ę
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
ť
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
J
0
1
2
3
$4
%5"
trackable_list_wrapper
<
0
1
$2
%3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
î
+trace_0
,trace_1
-trace_2
.trace_32
.__inference_sequential_12_layer_call_fn_955868
.__inference_sequential_12_layer_call_fn_956026
.__inference_sequential_12_layer_call_fn_956043
.__inference_sequential_12_layer_call_fn_955948Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 z+trace_0z,trace_1z-trace_2z.trace_3
Ú
/trace_0
0trace_1
1trace_2
2trace_32ď
I__inference_sequential_12_layer_call_and_return_conditional_losses_956072
I__inference_sequential_12_layer_call_and_return_conditional_losses_956115
I__inference_sequential_12_layer_call_and_return_conditional_losses_955967
I__inference_sequential_12_layer_call_and_return_conditional_losses_955986Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 z/trace_0z0trace_1z1trace_2z2trace_3
ćBă
!__inference__wrapped_model_955723!global_average_pooling2d_11_input"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 

3iter
	4decay
5learning_rate
6momentummomentumZmomentum[$momentum\%momentum]"
	optimizer
,
7serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object

=trace_02ă
<__inference_global_average_pooling2d_11_layer_call_fn_956120˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z=trace_0

>trace_02ţ
W__inference_global_average_pooling2d_11_layer_call_and_return_conditional_losses_956126˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z>trace_0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ŕ
Dtrace_0
Etrace_12Š
7__inference_batch_normalization_42_layer_call_fn_956139
7__inference_batch_normalization_42_layer_call_fn_956152´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 zDtrace_0zEtrace_1

Ftrace_0
Gtrace_12ß
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_956172
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_956206´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 zFtrace_0zGtrace_1
 "
trackable_list_wrapper
+:)2batch_normalization_42/gamma
*:(2batch_normalization_42/beta
3:1 (2"batch_normalization_42/moving_mean
7:5 (2&batch_normalization_42/moving_variance
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
í
Mtrace_02Đ
)__inference_dense_10_layer_call_fn_956215˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zMtrace_0

Ntrace_02ë
D__inference_dense_10_layer_call_and_return_conditional_losses_956226˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zNtrace_0
#:!
2dense_10/kernel
:2dense_10/bias
.
0
1"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_12_layer_call_fn_955868!global_average_pooling2d_11_input"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Bý
.__inference_sequential_12_layer_call_fn_956026inputs"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Bý
.__inference_sequential_12_layer_call_fn_956043inputs"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
.__inference_sequential_12_layer_call_fn_955948!global_average_pooling2d_11_input"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
I__inference_sequential_12_layer_call_and_return_conditional_losses_956072inputs"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
I__inference_sequential_12_layer_call_and_return_conditional_losses_956115inputs"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
śBł
I__inference_sequential_12_layer_call_and_return_conditional_losses_955967!global_average_pooling2d_11_input"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
śBł
I__inference_sequential_12_layer_call_and_return_conditional_losses_955986!global_average_pooling2d_11_input"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
ĺBâ
$__inference_signature_wrapper_956009!global_average_pooling2d_11_input"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
đBí
<__inference_global_average_pooling2d_11_layer_call_fn_956120inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
W__inference_global_average_pooling2d_11_layer_call_and_return_conditional_losses_956126inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ýBú
7__inference_batch_normalization_42_layer_call_fn_956139inputs"´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ýBú
7__inference_batch_normalization_42_layer_call_fn_956152inputs"´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_956172inputs"´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_956206inputs"´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_10_layer_call_fn_956215inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
řBő
D__inference_dense_10_layer_call_and_return_conditional_losses_956226inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
N
Q	variables
R	keras_api
	Stotal
	Tcount"
_tf_keras_metric
^
U	variables
V	keras_api
	Wtotal
	Xcount
Y
_fn_kwargs"
_tf_keras_metric
.
S0
T1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
:  (2total
:  (2count
.
W0
X1"
trackable_list_wrapper
-
U	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
6:42)SGD/batch_normalization_42/gamma/momentum
5:32(SGD/batch_normalization_42/beta/momentum
.:,
2SGD/dense_10/kernel/momentum
':%2SGD/dense_10/bias/momentumš
!__inference__wrapped_model_955723$%S˘P
I˘F
DA
!global_average_pooling2d_11_input˙˙˙˙˙˙˙˙˙
Ş "4Ş1
/
dense_10# 
dense_10˙˙˙˙˙˙˙˙˙ş
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_956172d4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ş
R__inference_batch_normalization_42_layer_call_and_return_conditional_losses_956206d4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
7__inference_batch_normalization_42_layer_call_fn_956139W4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙
7__inference_batch_normalization_42_layer_call_fn_956152W4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙Ś
D__inference_dense_10_layer_call_and_return_conditional_losses_956226^$%0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ~
)__inference_dense_10_layer_call_fn_956215Q$%0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙ŕ
W__inference_global_average_pooling2d_11_layer_call_and_return_conditional_losses_956126R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ˇ
<__inference_global_average_pooling2d_11_layer_call_fn_956120wR˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "!˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ű
I__inference_sequential_12_layer_call_and_return_conditional_losses_955967$%[˘X
Q˘N
DA
!global_average_pooling2d_11_input˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 Ű
I__inference_sequential_12_layer_call_and_return_conditional_losses_955986$%[˘X
Q˘N
DA
!global_average_pooling2d_11_input˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ż
I__inference_sequential_12_layer_call_and_return_conditional_losses_956072r$%@˘=
6˘3
)&
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ż
I__inference_sequential_12_layer_call_and_return_conditional_losses_956115r$%@˘=
6˘3
)&
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ł
.__inference_sequential_12_layer_call_fn_955868$%[˘X
Q˘N
DA
!global_average_pooling2d_11_input˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙ł
.__inference_sequential_12_layer_call_fn_955948$%[˘X
Q˘N
DA
!global_average_pooling2d_11_input˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
.__inference_sequential_12_layer_call_fn_956026e$%@˘=
6˘3
)&
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
.__inference_sequential_12_layer_call_fn_956043e$%@˘=
6˘3
)&
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙á
$__inference_signature_wrapper_956009¸$%x˘u
˘ 
nŞk
i
!global_average_pooling2d_11_inputDA
!global_average_pooling2d_11_input˙˙˙˙˙˙˙˙˙"4Ş1
/
dense_10# 
dense_10˙˙˙˙˙˙˙˙˙