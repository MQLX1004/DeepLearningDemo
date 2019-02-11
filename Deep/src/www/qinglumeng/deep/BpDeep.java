package www.qinglumeng.deep;

import java.util.Random;

public class BpDeep {
	public double[][] layer;//��¼��Ԫ���ֵ����һάΪ�������ڶ�άΪ�ò�ڼ���
	public double[][] layerErr;//��¼���ڵ����
	public double[][][] layerWeight;//��¼��Ԫ֮������Ȩ�أ���һάΪ��һ���ڵ�������ڶ�άΪ��һ���ڵ�ò�ڼ���������άΪ�ڶ����ڵ�����һ�εĵڼ���
	public double[][][] layerWeightDelta;//����ڵ�Ȩ�� ����
	public double mobp;//����ϵ��
	public double rate;//ѧϰϵ��
	
	public BpDeep(int[] layernum, double rate, double mobp){
		this.mobp = mobp;
		this.rate = rate;
		
		//�����������ʼ����һά
        layer = new double[layernum.length][];
        layerErr = new double[layernum.length][];
        layerWeight = new double[layernum.length][][];
        layerWeightDelta = new double[layernum.length][][];
        
        Random random = new Random();
        for(int l = 0; l < layernum.length; l++){
        	//�����������ʼ���ڶ�ά
            layer[l]=new double[layernum[l]];
            layerErr[l]=new double[layernum[l]];

                      
            if(l+1<layernum.length){//��Ϊ���һ������㲻��Ҫ������һ�㣬�����ж�
            	layerWeight[l]=new double[layernum[l]+1][layernum[l+1]];
            	layerWeightDelta[l]=new double[layernum[l]+1][layernum[l+1]];
            	
            	
                for(int j=0;j<layernum[l]+1;j++){
                	for(int i=0;i<layernum[l+1];i++){
                    	layerWeight[l][j][i]=random.nextDouble();//�����ʼ��Ȩ��
            		}
                }
               
            }
        }
	}
	public double[] computeOut(double[] in){
		for(int l = 1; l < layer.length; l++){//���Ʋ���
			for(int j = 0; j < layer[l].length; j++){//���Ƹò����
				double z = layerWeight[l-1][layer[l-1].length][j];//������¼�����ֵ
				for(int i = 0; i < layer[l-1].length; i++){//�����Ȩ����
                    layer[l-1][i]=l==1?in[i]:layer[l-1][i];
					//layer[l-1][i] = l == 1?in[i]:layer[l-1][i];//����ǵ�һ�㣬������ֵ������һ��
					z+=layerWeight[l-1][i][j] * layer[l-1][i];
				}
				layer[l][j] = 1/(1+Math.exp(-z));//ʹ��sigmoid�����
			}
		}
		
		return layer[layer.length-1];
	}
    //��㷴��������޸�Ȩ��
    public void updateWeight(double[] tar){
        int l=layer.length-1;
        for(int j=0;j<layerErr[l].length;j++)//�������һ�㣨����㣩���
            layerErr[l][j]=layer[l][j]*(1-layer[l][j])*(tar[j]-layer[l][j]);
        
        
 
        while(l-->0){//���������޸�Ȩ��
            for(int j=0;j<layerErr[l].length;j++){
				
                double z = 0.0;//zΪʵ����Ŀ��Ĳ��
				
                for(int i=0;i<layerErr[l+1].length;i++){
                    z=z+l>0?layerErr[l+1][i]*layerWeight[l][j][i]:0;//ʵ����Ŀ���ֵ
					
                    layerWeightDelta[l][j][i]= mobp*layerWeightDelta[l][j][i]+rate*layerErr[l+1][i]*layer[l][j];//�����㶯������
                    layerWeight[l][j][i]+=layerWeightDelta[l][j][i];//������Ȩ�ص���
                    
//					if(j==layerErr[l].length-1){
//						layerWeightDelta[l][j+1][i]= mobp*layerWeightDelta[l][j+1][i]+rate*layerErr[l+1][i];//�ؾද������
//						layerWeight[l][j+1][i]+=layerWeightDelta[l][j+1][i];//�ؾ�Ȩ�ص���
//                    }
                }
                
     
				
                layerErr[l][j]=z*layer[l][j]*(1-layer[l][j]);//�������
            }
        }
    }
    public void train(double[] in, double[] tar){
        double[] out = computeOut(in);
        updateWeight(tar);
    }
	
}
