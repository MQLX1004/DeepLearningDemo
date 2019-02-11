package www.qinglumeng.deep;

import java.util.Arrays;

public class BpDeepTest {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int[] in = new int[]{2,10,2};
		BpDeep bp = new BpDeep(in, 0.15, 0.8);//创建神经网络对象，第一个参数数组长度为层数，数组数值为每层个数，第二个参数为学习步长，第三个参数为动量系数
        //设置样本数据，对应上面的4个二维坐标数据
        double[][] data = new double[][]{{1,2},{2,2},{1,1},{2,1}};
        //设置目标数据，对应4个坐标数据的分类
        double[][] target = new double[][]{{1,0},{0,1},{0,1},{1,0}};
        //训练5000次
        for(int n=0;n<5000;n++)
            for(int i=0;i<data.length;i++)
                bp.train(data[i], target[i]);
        //根据训练结果来检验样本数据
        for(int j=0;j<data.length;j++){
            double[] result = bp.computeOut(data[j]);
            System.out.println(Arrays.toString(data[j])+":"+Arrays.toString(result));
        }
        //根据训练结果来预测一条新数据的分类
        double[] x = new double[]{4,1};
        double[] result = bp.computeOut(x);
        System.out.println(Arrays.toString(x)+":"+Arrays.toString(result));
	}

}
