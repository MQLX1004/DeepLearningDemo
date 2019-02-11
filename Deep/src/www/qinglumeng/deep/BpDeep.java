package www.qinglumeng.deep;

import java.util.Random;

public class BpDeep {
	public double[][] layer;//记录神经元结点值，第一维为层数，第二维为该层第几个
	public double[][] layerErr;//记录各节点误差
	public double[][][] layerWeight;//记录神经元之间链接权重，第一维为第一个节点层数，第二维为第一个节点该层第几个，第三维为第二个节点是下一次的第几个
	public double[][][] layerWeightDelta;//各层节点权重 动量
	public double mobp;//动量系数
	public double rate;//学习系数
	
	public BpDeep(int[] layernum, double rate, double mobp){
		this.mobp = mobp;
		this.rate = rate;
		
		//给各个数组初始化第一维
        layer = new double[layernum.length][];
        layerErr = new double[layernum.length][];
        layerWeight = new double[layernum.length][][];
        layerWeightDelta = new double[layernum.length][][];
        
        Random random = new Random();
        for(int l = 0; l < layernum.length; l++){
        	//给各个数组初始化第二维
            layer[l]=new double[layernum[l]];
            layerErr[l]=new double[layernum[l]];

                      
            if(l+1<layernum.length){//因为最后一层输出层不需要链接下一层，故做判断
            	layerWeight[l]=new double[layernum[l]+1][layernum[l+1]];
            	layerWeightDelta[l]=new double[layernum[l]+1][layernum[l+1]];
            	
            	
                for(int j=0;j<layernum[l]+1;j++){
                	for(int i=0;i<layernum[l+1];i++){
                    	layerWeight[l][j][i]=random.nextDouble();//随机初始化权重
            		}
                }
               
            }
        }
	}
	public double[] computeOut(double[] in){
		for(int l = 1; l < layer.length; l++){//控制层数
			for(int j = 0; j < layer[l].length; j++){//控制该层个数
				double z = layerWeight[l-1][layer[l-1].length][j];//用来记录计算的值
				for(int i = 0; i < layer[l-1].length; i++){//逐个加权计算
                    layer[l-1][i]=l==1?in[i]:layer[l-1][i];
					//layer[l-1][i] = l == 1?in[i]:layer[l-1][i];//如果是第一层，将输入值赋给第一层
					z+=layerWeight[l-1][i][j] * layer[l-1][i];
				}
				layer[l][j] = 1/(1+Math.exp(-z));//使用sigmoid激活函数
			}
		}
		
		return layer[layer.length-1];
	}
    //逐层反向计算误差并修改权重
    public void updateWeight(double[] tar){
        int l=layer.length-1;
        for(int j=0;j<layerErr[l].length;j++)//计算最后一层（输出层）误差
            layerErr[l][j]=layer[l][j]*(1-layer[l][j])*(tar[j]-layer[l][j]);
        
        
 
        while(l-->0){//逐层计算误差并修改权重
            for(int j=0;j<layerErr[l].length;j++){
				
                double z = 0.0;//z为实际与目标的差距
				
                for(int i=0;i<layerErr[l+1].length;i++){
                    z=z+l>0?layerErr[l+1][i]*layerWeight[l][j][i]:0;//实际与目标差值
					
                    layerWeightDelta[l][j][i]= mobp*layerWeightDelta[l][j][i]+rate*layerErr[l+1][i]*layer[l][j];//隐含层动量调整
                    layerWeight[l][j][i]+=layerWeightDelta[l][j][i];//隐含层权重调整
                    
//					if(j==layerErr[l].length-1){
//						layerWeightDelta[l][j+1][i]= mobp*layerWeightDelta[l][j+1][i]+rate*layerErr[l+1][i];//截距动量调整
//						layerWeight[l][j+1][i]+=layerWeightDelta[l][j+1][i];//截距权重调整
//                    }
                }
                
     
				
                layerErr[l][j]=z*layer[l][j]*(1-layer[l][j]);//计算误差
            }
        }
    }
    public void train(double[] in, double[] tar){
        double[] out = computeOut(in);
        updateWeight(tar);
    }
	
}
