
def tt02():
    print("tt02")

def run():
    print("Starting...")

    model, processor, device = load_model()
    video = preprocess(video)

    print("Finished.")

if __name__ == "__main__":
    run()
    tt02()
    print("Done.")