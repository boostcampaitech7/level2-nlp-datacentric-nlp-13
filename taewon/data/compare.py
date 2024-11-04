import pandas as pd
import argparse

def compare_csv_files(file1_path, file2_path, output_path):
    # CSV 파일 읽기
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # 'ID'와 'target' 열이 있는지 확인
    required_columns = ['ID', 'target']
    for df in [df1, df2]:
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV 파일에 'ID'와 'target' 열이 모두 있어야 합니다.")

    # 두 DataFrame을 'ID'를 기준으로 병합
    merged_df = pd.merge(df1[['ID', 'target']], df2[['ID', 'target']], on='ID', suffixes=('_1', '_2'))

    # target 값이 다른 행 찾기
    diff_rows = merged_df[merged_df['target_1'] != merged_df['target_2']]

    # 결과 DataFrame 생성
    result_df = pd.DataFrame({
        'ID': diff_rows['ID'],
        'target_file1': diff_rows['target_1'],
        'target_file2': diff_rows['target_2']
    })

    # 결과를 CSV 파일로 저장
    result_df.to_csv(output_path, index=False)
    print(f"결과가 {output_path}에 저장되었습니다.")
    print(f"총 {len(result_df)}개의 다른 target 값이 발견되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="두 CSV 파일에서 ID별 target 값이 다른 행을 찾습니다.")
    parser.add_argument("file1", help="첫 번째 CSV 파일 경로")
    parser.add_argument("file2", help="두 번째 CSV 파일 경로")
    parser.add_argument("output", help="결과를 저장할 CSV 파일 경로")
    
    args = parser.parse_args()

    compare_csv_files(args.file1, args.file2, args.output)