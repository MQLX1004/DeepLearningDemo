package www.qinglumeng.deep;

import java.util.Arrays;

public class BpDeepTest {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int[] in = new int[]{2,10,2};
		BpDeep bp = new BpDeep(in, 0.15, 0.8);//������������󣬵�һ���������鳤��Ϊ������������ֵΪÿ��������ڶ�������Ϊѧϰ����������������Ϊ����ϵ��
        //�����������ݣ���Ӧ�����4����ά��������
        double[][] data = new double[][]{{1,2},{2,2},{1,1},{2,1}};
        //����Ŀ�����ݣ���Ӧ4���������ݵķ���
        double[][] target = new double[][]{{1,0},{0,1},{0,1},{1,0}};
        //ѵ��5000��
        for(int n=0;n<5000;n++)
            for(int i=0;i<data.length;i++)
                bp.train(data[i], target[i]);
        //����ѵ�������������������
        for(int j=0;j<data.length;j++){
            double[] result = bp.computeOut(data[j]);
            System.out.println(Arrays.toString(data[j])+":"+Arrays.toString(result));
        }
        //����ѵ�������Ԥ��һ�������ݵķ���
        double[] x = new double[]{4,1};
        double[] result = bp.computeOut(x);
        System.out.println(Arrays.toString(x)+":"+Arrays.toString(result));
	}

}
