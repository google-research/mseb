Ȝ:�
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
identifiers
_identifiers

candidates	
_candidates	
query_with_exclusions


signatures"
_tf_keras_model
.
0
1	"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_02�
+__inference_brute_force_1_layer_call_fn_122�*
 
����
	jqueries
jk
args
 
varargs
 
varkw�

 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpecztrace_0
�
trace_02�
F__inference_brute_force_1_layer_call_and_return_conditional_losses_111�*
 
����
	jqueries
jk
args
 
varargs
 
varkw�

 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpecztrace_0
�B�
__inference__wrapped_model_97input_1"�*
 
����

jargs_0
args
 
varargs
 
varkw
 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec
:2identifiers
:2
candidates
�2��*
 
���)!�
	jqueries
j
exclusions
jk
args
 
varargs
 
varkw�

 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec
,
serving_default"
signature_map
.
0
1	"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_brute_force_1_layer_call_fn_122input_1"�*
 
����
	jqueries
jk
args
 
varargs
 
varkw
 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec
�B�
F__inference_brute_force_1_layer_call_and_return_conditional_losses_111input_1"�*
 
����
	jqueries
jk
args
 
varargs
 
varkw
 
defaults� 

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec
�B�
!__inference_signature_wrapper_134input_1"�*
 
���� 
args
 
varargs
 
varkw
 
defaults�
	jinput_1

kwonlyargs
 
kwonlydefaults� 
annotations
FullArgSpec�
__inference__wrapped_model_97�	0�-
&�#
!�
input_1���������
� "c�`
.
output_1"�
output_1���������
.
output_2"�
output_2����������
F__inference_brute_force_1_layer_call_and_return_conditional_losses_111�	4�1
*�'
!�
input_1���������

 
� "Y�V
O�L
$�!

tensor_0_0���������
$�!

tensor_0_1���������
� �
+__inference_brute_force_1_layer_call_fn_122�	4�1
*�'
!�
input_1���������

 
� "K�H
"�
tensor_0���������
"�
tensor_1����������
!__inference_signature_wrapper_134�	;�8
� 
1�.
,
input_1!�
input_1���������"c�`
.
output_1"�
output_1���������
.
output_2"�
output_2���������*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������<
output_20
StatefulPartitionedCall:1���������tensorflow/serving/predict"
saved_model_main_op

NoOpL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28�n
�

candidatesVarHandleOp*
_output_shapes
: *

debug_namecandidates/*
dtype0*
shape
:*
shared_name
candidates
i
candidates/Read/ReadVariableOpReadVariableOp
candidates*
_output_shapes

:*
dtype0
�
identifiersVarHandleOp*
_output_shapes
: *

debug_nameidentifiers/*
dtype0*
shape:*
shared_nameidentifiers
g
identifiers/Read/ReadVariableOpReadVariableOpidentifiers*
_output_shapes
:*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1
candidatesidentifiers*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*$
_read_only_resource_inputs
*2
config_proto" �J � 82J 

CPU

GPU **
f%R#
!__inference_signature_wrapper_134

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
identifiers
_identifiers

candidates	
_candidates	
query_with_exclusions


signatures*

0
1	*
* 
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
KE
VARIABLE_VALUEidentifiers&identifiers/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUE
candidates%candidates/.ATTRIBUTES/VARIABLE_VALUE*
* 

serving_default* 

0
1	*
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameidentifiers
candidatesConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" �J � 82J 

CPU

GPU *%
f R
__inference__traced_save_169
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameidentifiers
candidates*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" �J � 82J 

CPU

GPU *(
f#R!
__inference__traced_restore_184�Y
�:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:+'
%
_user_specified_nameidentifiers:*&
$
_user_specified_name
candidates:=9

_output_shapes
: 

_user_specified_nameConst2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : "!

identity_5Identity_5:output:0w
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
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: e
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_identifiers*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_identifiers^Read/DisableCopyOnRead*
_output_shapes
:*
dtype0V
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:]

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:h
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_candidates*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_candidates^Read_1/DisableCopyOnRead*
_output_shapes

:*
dtype0^

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:c

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes

:L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
valuewBuB&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHs
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h

Identity_4Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: S

Identity_5IdentityIdentity_4:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp*
_output_shapes
 
�
__inference__traced_save_169
file_prefix0
"read_disablecopyonread_identifiers:5
#read_1_disablecopyonread_candidates:
savev2_const

identity_5��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp
�:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource2,
brute_force_1/Gatherbrute_force_1/Gather2J
#brute_force_1/MatMul/ReadVariableOp#brute_force_1/MatMul/ReadVariableOp*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : "
identityIdentity:output:0"!

identity_1Identity_1:output:0�
#brute_force_1/MatMul/ReadVariableOpReadVariableOp,brute_force_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
brute_force_1/MatMulMatMulinput_1+brute_force_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*
transpose_b(X
brute_force_1/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :�
brute_force_1/TopKV2TopKV2brute_force_1/MatMul:product:0brute_force_1/TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������:����������
brute_force_1/GatherResourceGatherbrute_force_1_gather_resourcebrute_force_1/TopKV2:indices:0*
Tindices0*'
_output_shapes
:���������*
dtype0l
IdentityIdentitybrute_force_1/TopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������n

Identity_1Identitybrute_force_1/Gather:output:0^NoOp*
T0*'
_output_shapes
:���������_
NoOpNoOp^brute_force_1/Gather$^brute_force_1/MatMul/ReadVariableOp*
_output_shapes
 
�
__inference__wrapped_model_97
input_1>
,brute_force_1_matmul_readvariableop_resource:+
brute_force_1_gather_resource:
identity

identity_1��brute_force_1/Gather�#brute_force_1/MatMul/ReadVariableOp
�:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:+'
%
_user_specified_nameidentifiers:*&
$
_user_specified_name
candidates2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1*(
_construction_contextkEagerRuntime*
_input_shapes
: : : "!

identity_3Identity_3:output:0�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
valuewBuB&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHv
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_identifiersIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_candidatesIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_3IdentityIdentity_2:output:0^NoOp_1*
T0*
_output_shapes
: L
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1*
_output_shapes
 
�
__inference__traced_restore_184
file_prefix*
assignvariableop_identifiers:/
assignvariableop_1_candidates:

identity_3��AssignVariableOp�AssignVariableOp_1
�	:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:#

_user_specified_name114:#

_user_specified_name11622
StatefulPartitionedCallStatefulPartitionedCall*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : "
identityIdentity:output:0"!

identity_1Identity_1:output:0�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*$
_read_only_resource_inputs
*2
config_proto" �J � 82J 

CPU

GPU *O
fJRH
F__inference_brute_force_1_layer_call_and_return_conditional_losses_111o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 
�
+__inference_brute_force_1_layer_call_fn_122
input_1
unknown:
	unknown_0:
identity

identity_1��StatefulPartitionedCall
�:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : "
identityIdentity:output:0"!

identity_1Identity_1:output:0t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0}
MatMulMatMulinput_1MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������:����������
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:���������*
dtype0^
IdentityIdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������`

Identity_1IdentityGather:output:0^NoOp*
T0*'
_output_shapes
:���������C
NoOpNoOp^Gather^MatMul/ReadVariableOp*
_output_shapes
 
�
F__inference_brute_force_1_layer_call_and_return_conditional_losses_111
input_10
matmul_readvariableop_resource:
gather_resource:
identity

identity_1��Gather�MatMul/ReadVariableOp
�:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:#

_user_specified_name126:#

_user_specified_name12822
StatefulPartitionedCallStatefulPartitionedCall*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : "
identityIdentity:output:0"!

identity_1Identity_1:output:0�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*$
_read_only_resource_inputs
*2
config_proto" �J � 82J 

CPU

GPU *&
f!R
__inference__wrapped_model_97o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 
�
!__inference_signature_wrapper_134
input_1
unknown:
	unknown_0:
identity

identity_1��StatefulPartitionedCall"�
�82unknown*2.20.0"serve�
^
AssignVariableOp
resource
value"dtype"type
dtype"( bool
validate_shape�
8
Const
output"dtype"tensor
value"type
dtype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	type
T
�
MatMul
a"T
b"T
product"T"( bool
transpose_a"( bool
transpose_b":
2	type
T"( bool
grad_a"( bool
grad_b
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"(bool
delete_old_dirs"( bool
allow_missing_files�

NoOp
M
Pack
values"T*N
output"T"0(int
N"	type
T" int
axis
C
Placeholder
output"dtype"type
dtype":shape
shape
@
ReadVariableOp
resource
value"dtype"type
dtype�
�
ResourceGather
resource
indices"Tindices
output"dtype" int

batch_dims"(bool
validate_indices"type
dtype":
2	type
Tindices�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"0(
list(type)
dtypes�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"0(
list(type)
dtypes�
?
Select
	condition

t"T
e"T
output"T"	type
T
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"(
list(type)
Tin"(
list(type)
Tout"	func
f" string
config" string
config_proto" string
executor_type��
@
StaticRegexFullMatch	
input

output
"string
pattern
L

StringJoin
inputs*N

output"
(int
N" string
	separator
�
TopKV2

input"T
k"Tk
values"T
indices"
index_type"(bool
sorted":
2	type
T":
2	0type
Tk":
2	0type

index_type
�
VarHandleOp
resource" string
	container" string
shared_name" string

debug_name"type
dtype"shape
shape"#
 list(string)
allowed_devices�