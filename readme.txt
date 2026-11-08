<src>文件夹中GenCase.cu文件为<基于亚塑性本构的有限土体刚性挡墙主动土压力SPH模拟研究>稿件中挡土墙算例模型
<src>文件夹中其余文件为相关计算程序
注：
①墙土摩擦角在particle_interaction.cu的cuParticleInteractionMaterialLoopContact()中更改，
②cuParticleInteractionMaterialLoopSolution()中contact_shift_i定义接触面是否发生切向滑移，并设定合理的速度场。
