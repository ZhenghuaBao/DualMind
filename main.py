import argparse
from pathlib import Path

if __name__=='__main__':
    args = argparse.ArgumentParser(description='Generate 5 pillars plus answers with VLM and LLM.')

    args.add_argument('--vlm', type=str, default='', help='specify the VLM to use for the forgery detection')
    args.add_argument('--llm', type=str, default='', help='specify the LLM to use for the answer generation')
    args.add_argument('--vlm_story', type=str, default='', help='specify the VLM to use for the STORY generation')

    args.add_argument('--ris', type=str, default='', help='specify the RIS to use for revers image search')
    args.add_argument('--ranking', type=str, default='', help='specify the ranking mechanism')
    args.add_argument('--credibility', type=str, default='', help='specify the credibility check')

    args.add_argument('--image_path', type=str, default='/dataset/img', help='specify the path to the image to be processed')
    args.add_argument('--output_path', type=str, default='/output', help='specify the path to store the output results')

    vlm = args.vlm
    vlm_story = args.vlm_story
    

    llm = args.llm
    
    image_path = Path(args.image_path)
    output_path = Path(args.output_path)


    for img in image_path.iterdir():
        _, img_exp, manipulated = vlm(img)
        story = vlm_story(img)


        img_ris_img, img_ris_txt = RIS(img)

        # TODO: using keywords probably not that ideal for web search
        # instead simply use GPT to formulate the current STORY into a suitable query and use it for search
        story_img, story_txt = keyword_search(story)

        
        match_img_ris_img, match_img_ris_txt = RTE(img_ris_txt, story)


        core_answers = generate_5PilsPlus(match_img_ris_img, match_img_ris_txt, img_exp)

        if similar(img, story_img):
            fallback_answers = generate_5PilsPlus(story_img, story_txt)
        else:
            story_exp = llm_similar(story, story_txt)
            fallback_answers = generate_5PilsPlus(story_img, story_txt)

        if sufficient(core_answers):
            # still store the fallback answers for evaluation
            # mark that the core answers are used to generate the final 5Pils+ answers
            # NOTE: we may simply conbine both pipelines to evluate if there is any performance gain
            save_results(core_answers, fallback_answers, 'core', 'output/results.json')
        else:
            save_results(core_answers, fallback_answers, 'fallback', 'output/results.json')
        print("Results saved successfully") 

    
    

    